"""Fixed Haar image pyramid + frozen DINO, fused before expert gaze.

One topology for both datasets. Wavelet filtering, decimation, amplitude scaling,
DINO spatial resizing and gaze pooling have no learned parameters.
"""
import torch
from torch import nn
import torch.nn.functional as F


class FixedHaarPyramid(nn.Module):
    def __init__(self):
        super().__init__()
        # Explicit band convention: LL, LH (column differences), HL (row
        # differences), HH. Sign conventions vary across wavelet libraries.
        kernels=torch.tensor([[[1.,1.],[1.,1.]],[[1.,-1.],[1.,-1.]],
                              [[1.,1.],[-1.,-1.]],[[1.,-1.],[-1.,1.]]])[:,None]/2
        self.register_buffer('kernels',kernels)

    def decompose(self,x):
        if x.ndim!=4 or x.shape[-1]%2 or x.shape[-2]%2:
            raise ValueError('Haar decomposition requires an even spatial grid.')
        bands=F.conv2d(x,self.kernels.to(dtype=x.dtype).repeat(x.shape[1],1,1,1),stride=2,groups=x.shape[1])
        return bands.reshape(x.shape[0],x.shape[1],4,*bands.shape[-2:])

    def forward(self,image):
        if image.ndim!=4 or image.shape[1]!=3 or any(s%16 for s in image.shape[-2:]):
            raise ValueError('Expected RGB with spatial dimensions divisible by 16.')
        x=image.float();pyramid=[]
        for level in range(1,5):
            bands=self.decompose(x)
            # At every level recurse only through raw LL, keeping RGB channels.
            # Return all four bands, with fixed 2^-level amplitude normalization.
            w=bands.permute(0,2,1,3,4).reshape(image.shape[0],12,*bands.shape[-2:])/(2**level)
            pyramid.append(w);x=bands[:,:,0]
        return pyramid


class ConvBlock(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.GELU(),
                         nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.GELU())


def prototype_contrast(z,h,eps=1e-6):
    tokens=z.flatten(2);fg_weights=h.flatten(2);bg_weights=1-fg_weights
    fg_mass=fg_weights.sum(-1,keepdim=True);bg_mass=bg_weights.sum(-1,keepdim=True)
    fg=(tokens*fg_weights).sum(-1,keepdim=True)/fg_mass.clamp_min(eps)
    bg=(tokens*bg_weights).sum(-1,keepdim=True)/bg_mass.clamp_min(eps)
    tokens=F.normalize(tokens,dim=1,eps=eps)
    sfg=(tokens*F.normalize(fg,dim=1,eps=eps)).sum(1,keepdim=True)
    sbg=(tokens*F.normalize(bg,dim=1,eps=eps)).sum(1,keepdim=True)
    valid=(fg_mass>eps)&(bg_mass>eps)
    return ((sfg-sbg).relu()*valid.to(z.dtype)).reshape(z.shape[0],1,*z.shape[-2:])


class WaveletGazeUNet(nn.Module):
    def __init__(self):
        super().__init__();widths=[32,64,128,256]
        self.wavelet=FixedHaarPyramid()
        self.dino_projection=nn.ModuleList([nn.Sequential(nn.Conv2d(1024,64,1,bias=False),
                                                       nn.GroupNorm(8,64),nn.GELU()) for _ in widths])
        self.feature_fusion=nn.ModuleList([nn.Sequential(nn.Conv2d(12+64,c,1,bias=False),
                                                      nn.GroupNorm(8,c),nn.GELU()) for c in widths])
        self.gaze_conditioning=nn.ModuleList([nn.Sequential(nn.Conv2d(c+2,c,1,bias=False),
                                                         nn.GroupNorm(8,c),nn.GELU()) for c in widths])
        self.bottleneck=ConvBlock(256,256)
        self.decoder=nn.ModuleList([ConvBlock(384,128),ConvBlock(192,64),ConvBlock(96,32),ConvBlock(35,16)])
        self.mask_head=nn.Conv2d(16,1,1)
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
                if layer.bias is not None:nn.init.zeros_(layer.bias)
        nn.init.normal_(self.mask_head.weight,std=.01)

    def forward(self,image,features,heatmap_full,return_intermediates=False):
        if image.ndim!=4 or image.shape[1]!=3 or any(s%16 for s in image.shape[-2:]):
            raise ValueError('Input must be Bx3xHxW with H and W divisible by 16.')
        if heatmap_full.shape!=(image.shape[0],1,*image.shape[-2:]):
            raise ValueError('Heatmap must be built at input image resolution.')
        if not torch.isfinite(heatmap_full).all() or heatmap_full.min()<0 or heatmap_full.max()>1:
            raise ValueError('Expected a finite heatmap in [0,1].')
        if len(features)!=4 or any(f.ndim!=4 or f.shape[:2]!=(image.shape[0],1024) for f in features):
            raise ValueError('Expected four frozen 1024-channel DINO depth features.')
        pyramid=self.wavelet(image);skips=[];trace=[]
        for w,f,project,fuse,condition in zip(pyramid,features,self.dino_projection,self.feature_fusion,self.gaze_conditioning):
            p=F.interpolate(project(f.detach().float()),size=w.shape[-2:],mode='bilinear',align_corners=False)
            z=fuse(torch.cat([w,p],dim=1))
            h=F.adaptive_avg_pool2d(heatmap_full.float(),w.shape[-2:])
            c=prototype_contrast(z,h);s=condition(torch.cat([z,h,c],dim=1));skips.append(s)
            if return_intermediates:trace.append(dict(wavelet=w,dino=p,fused=z,gaze=h,contrast=c,skip=s))
        x=self.bottleneck(skips[3])
        for decode,skip in zip(self.decoder,[skips[2],skips[1],skips[0],image.float()]):
            x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=decode(torch.cat([x,skip],dim=1))
        logits=self.mask_head(x)
        return (logits,trace) if return_intermediates else logits


def segmentation_loss(logits,target):
    p=logits.sigmoid();axes=(1,2,3)
    soft_dice=1-((2*(p*target).sum(axes)+1e-6)/(p.sum(axes)+target.sum(axes)+1e-6)).mean()
    return F.binary_cross_entropy_with_logits(logits,target)+soft_dice
