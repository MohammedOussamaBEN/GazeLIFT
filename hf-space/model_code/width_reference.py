"""32-channel DINO projections and four32-channel gaze-conditioned skips."""
from torch import nn
from reference_model import DinoGazeWaveletDecoder
from base_model import ConvBlock,segmentation_loss


class Dino32Gaze32WaveletDecoder(DinoGazeWaveletDecoder):
    def __init__(self):
        # Same-shaped retained modules start exactly as the baseline at seed42.
        super().__init__()
        self.dino_projection=nn.ModuleList([nn.Sequential(nn.Conv2d(1024,32,1,bias=False),nn.GroupNorm(8,32),nn.GELU()) for _ in range(4)])
        self.gaze_conditioning=nn.ModuleList([nn.Sequential(nn.Conv2d(34,32,1,bias=False),nn.GroupNorm(8,32),nn.GELU()) for _ in range(4)])
        self.deep_wavelet_fusion=nn.Sequential(nn.Conv2d(44,256,1,bias=False),nn.GroupNorm(8,256),nn.GELU())
        self.decoder[0]=ConvBlock(300,128)
        self.decoder[1]=ConvBlock(172,64)
        for group in [self.dino_projection,self.gaze_conditioning,self.deep_wavelet_fusion,self.decoder[0],self.decoder[1]]:
            for layer in group.modules():
                if isinstance(layer,nn.Conv2d):nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
