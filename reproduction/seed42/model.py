"""Add one learned spatial3x3 Conv2d after each transposed decoder block."""
from torch import nn
from transpose_reference import TransposeGazePyramidDecoder
from base_model import segmentation_loss

class TransposeCNNGazePyramidDecoder(TransposeGazePyramidDecoder):
    def __init__(self):
        super().__init__()
        for block in self.decoder:
            conv=nn.Conv2d(32,32,3,padding=1,bias=False)
            nn.init.kaiming_normal_(conv.weight,mode='fan_out',nonlinearity='relu')
            block.add_module('3',conv)
            block.add_module('4',nn.GroupNorm(8,32))
            block.add_module('5',nn.GELU())
