"""Pyramid DINO projections with single 3x3 decoder convolutions."""
from torch import nn
from pyramid_reference import PyramidPointwiseRGBWaveletDecoder
from base_model import segmentation_loss

class PyramidConv3RGBWaveletDecoder(PyramidPointwiseRGBWaveletDecoder):
    def __init__(self):
        super().__init__()
        self.decoder=nn.ModuleList([nn.Sequential(nn.Conv2d(cin,32,3,padding=1,bias=False),nn.GroupNorm(8,32),nn.GELU()) for cin in [44,44,44,44,35]])
        for layer in self.decoder.modules():
            if isinstance(layer,nn.Conv2d):nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
