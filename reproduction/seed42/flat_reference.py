"""Flat32 decoder: each block has exactly one44->32 spatial convolution."""
import torch
from torch import nn
import torch.nn.functional as F
from width_reference import Dino32Gaze32WaveletDecoder
from base_model import prototype_contrast,segmentation_loss


class Flat32SingleConvWaveletDecoder(Dino32Gaze32WaveletDecoder):
    def __init__(self):
        super().__init__()
        # No wide bottleneck, no separate deep projection, no retained double blocks.
        del self.deep_wavelet_fusion
        del self.bottleneck
        self.decoder=nn.ModuleList([nn.Sequential(nn.Conv2d(44,32,3,padding=1,bias=False),nn.GroupNorm(8,32),nn.GELU()) for _ in range(5)])
        self.mask_head=nn.Conv2d(32,1,1)
        for group in [self.decoder,self.mask_head]:
            for layer in group.modules():
                if isinstance(layer,nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
                    if layer.bias is not None:nn.init.zeros_(layer.bias)
        nn.init.normal_(self.mask_head.weight,std=.01)

    def forward(self,image,features,heatmap,return_intermediates=False):
        if image.ndim!=4 or image.shape[1]!=3 or any(s%16 for s in image.shape[-2:]):raise ValueError('Expected RGB, dimensions divisible by16')
        if heatmap.shape!=(image.shape[0],1,*image.shape[-2:]):raise ValueError('Heatmap must match input size')
        if len(features)!=4 or any(f.ndim!=4 or f.shape[:2]!=(image.shape[0],1024) for f in features):raise ValueError('Expected four frozen1024-channel DINO maps')
        sizes=[(image.shape[-2]//s,image.shape[-1]//s) for s in [2,4,8,16]]
        skips=[];trace=[]
        for f,size,project,condition in zip(features,sizes,self.dino_projection,self.gaze_conditioning):
            p=F.interpolate(project(f.detach().float()),size=size,mode='bilinear',align_corners=False)
            h=F.adaptive_avg_pool2d(heatmap.float(),size);c=prototype_contrast(p,h)
            s=condition(torch.cat([p,h,c],1));skips.append(s)
            if return_intermediates:trace.append(dict(dino=p,gaze=h,contrast=c,conditioned=s))
        waves=self.wavelet(image);decoder_trace=[]
        z=torch.cat([skips[3],waves[3]],1);x=self.decoder[0](z)
        if return_intermediates:decoder_trace.append(dict(level=3,input=z,output=x))
        for decoder,level in zip(self.decoder[1:4],[2,1,0]):
            merged=F.interpolate(x,size=sizes[level],mode='bilinear',align_corners=False)+skips[level]
            z=torch.cat([merged,waves[level]],1);x=decoder(z)
            if return_intermediates:decoder_trace.append(dict(level=level,input=z,output=x,merged=merged))
        up=F.interpolate(x,size=image.shape[-2:],mode='bilinear',align_corners=False)
        wave_full=F.interpolate(waves[0],size=image.shape[-2:],mode='bilinear',align_corners=False)
        z=torch.cat([up,wave_full],1);x=self.decoder[4](z)
        if return_intermediates:decoder_trace.append(dict(level='full',input=z,output=x))
        logits=self.mask_head(x)
        return (logits,trace,waves,decoder_trace) if return_intermediates else logits
