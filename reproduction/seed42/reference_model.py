"""Condition DINO alone with gaze; introduce wavelets only inside decoding."""
import torch
from torch import nn
import torch.nn.functional as F
from base_model import ConvBlock,prototype_contrast,segmentation_loss
from wavelet import WaveletPool


class DinoGazeWaveletDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.wavelet=WaveletPool();widths=[32,64,128,256]
        self.dino_projection=nn.ModuleList([nn.Sequential(nn.Conv2d(1024,64,1,bias=False),nn.GroupNorm(8,64),nn.GELU()) for _ in widths])
        self.gaze_conditioning=nn.ModuleList([nn.Sequential(nn.Conv2d(66,c,1,bias=False),nn.GroupNorm(8,c),nn.GELU()) for c in widths])
        self.deep_wavelet_fusion=nn.Sequential(nn.Conv2d(268,256,1,bias=False),nn.GroupNorm(8,256),nn.GELU())
        self.bottleneck=ConvBlock(256,256)
        self.decoder=nn.ModuleList([ConvBlock(a,b) for a,b in zip([396,204,108,35],[128,64,32,16])])
        self.mask_head=nn.Conv2d(16,1,1)
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
                if layer.bias is not None:nn.init.zeros_(layer.bias)
        nn.init.normal_(self.mask_head.weight,std=.01)

    def forward(self,image,features,heatmap,return_intermediates=False):
        if image.ndim!=4 or image.shape[1]!=3 or any(s%16 for s in image.shape[-2:]):raise ValueError('Expected RGB, dimensions divisible by16')
        if heatmap.shape!=(image.shape[0],1,*image.shape[-2:]):raise ValueError('Heatmap must match input size')
        if len(features)!=4 or any(f.ndim!=4 or f.shape[:2]!=(image.shape[0],1024) for f in features):raise ValueError('Expected four1024-channel frozen DINO features')
        sizes=[(image.shape[-2]//s,image.shape[-1]//s) for s in [2,4,8,16]]
        skips=[];trace=[]
        # This entire path is independent of the image/wavelet branch.
        for f,size,project,condition in zip(features,sizes,self.dino_projection,self.gaze_conditioning):
            p=F.interpolate(project(f.detach().float()),size=size,mode='bilinear',align_corners=False)
            h=F.adaptive_avg_pool2d(heatmap.float(),size);c=prototype_contrast(p,h)
            s=condition(torch.cat([p,h,c],1));skips.append(s)
            if return_intermediates:trace.append(dict(dino=p,gaze=h,contrast=c,conditioned=s))
        waves=self.wavelet(image)
        x=self.bottleneck(self.deep_wavelet_fusion(torch.cat([skips[3],waves[3]],1)))
        for decoder,level in zip(self.decoder[:3],[2,1,0]):
            x=F.interpolate(x,size=sizes[level],mode='bilinear',align_corners=False)
            x=decoder(torch.cat([x,skips[level],waves[level]],1))
        x=self.decoder[3](torch.cat([F.interpolate(x,size=image.shape[-2:],mode='bilinear',align_corners=False),image.float()],1))
        logits=self.mask_head(x)
        return (logits,trace,waves) if return_intermediates else logits
