"""DINO projections 32/64/128/256; gaze outputs and pointwise decoder remain32."""
from torch import nn
from pointwise_reference import PointwiseRGBWaveletDecoder
from base_model import segmentation_loss

class PyramidPointwiseRGBWaveletDecoder(PointwiseRGBWaveletDecoder):
    def __init__(self):
        super().__init__()
        # Preserve the unchanged first scale and all wavelet/decoder/head initial values.
        for i,width in enumerate([64,128,256],start=1):
            self.dino_projection[i]=nn.Sequential(nn.Conv2d(1024,width,1,bias=False),nn.GroupNorm(8,width),nn.GELU())
            self.gaze_conditioning[i]=nn.Sequential(nn.Conv2d(width+2,32,1,bias=False),nn.GroupNorm(8,32),nn.GELU())
            for module in [self.dino_projection[i],self.gaze_conditioning[i]]:
                for layer in module.modules():
                    if isinstance(layer,nn.Conv2d):nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
