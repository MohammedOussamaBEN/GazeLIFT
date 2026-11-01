"""Exact standalone lifting-wavelet classes from previous experiment."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class LiftingAxis(nn.Module):
    """Shared linear predict/update filters, initially the Haar transform.

    Each three-tap predictor sums to one; each updater sums to one half. Two
    independent coefficients per filter give four trainable scalars per axis.
    Replicate extension preserves constants at boundaries. Lifting is invertible
    for every finite coefficient setting; it is not generally orthogonal after
    learning. No perfect-reconstruction claim is made for subsequent pooling.
    """

    def __init__(self, axis):
        super().__init__()
        if axis not in (-2, -1):
            raise ValueError("The lifting axis must be height or width.")
        self.axis = axis
        self.predict_edges = nn.Parameter(torch.zeros(2))
        self.update_edges = nn.Parameter(torch.zeros(2))

    def filters(self):
        p, u = self.predict_edges, self.update_edges
        predict = torch.stack((p[0], 1 - p.sum(), p[1]))
        update = torch.stack((u[0], .5 - u.sum(), u[1]))
        return predict, update

    def _filter(self, value, kernel):
        # Move the active spatial axis to width, filter each channel identically.
        value = value.movedim(self.axis, -1)
        channels = value.shape[1]
        weights = kernel.to(value.dtype).reshape(1, 1, 1, 3).expand(channels, 1, 1, 3)
        filtered = F.conv2d(F.pad(value, (1, 1, 0, 0), mode="replicate"),
                            weights, groups=channels)
        return filtered.movedim(-1, self.axis)

    def forward(self, value):
        if value.shape[self.axis] % 2:
            raise ValueError("Lifting requires an even active dimension.")
        moved = value.movedim(self.axis, -1)
        even = moved[..., ::2].movedim(-1, self.axis)
        odd = moved[..., 1::2].movedim(-1, self.axis)
        predict, update = self.filters()
        detail = odd - self._filter(even, predict)
        smooth = even + self._filter(detail, update)
        # Haar signs/scaling match the fixed reference at initialization.
        return smooth * math.sqrt(2), -detail / math.sqrt(2)

    def inverse(self, low, high):
        if low.shape != high.shape:
            raise ValueError("Low and high coefficients must have equal shapes.")
        predict, update = self.filters()
        detail = -high * math.sqrt(2)
        even = low / math.sqrt(2) - self._filter(detail, update)
        odd = detail + self._filter(even, predict)
        interleaved = torch.stack((even.movedim(self.axis, -1),
                                   odd.movedim(self.axis, -1)), dim=-1)
        return interleaved.flatten(-2, -1).movedim(-1, self.axis)

class LearnableWavelet2D(nn.Module):
    """One separable 2-D analysis, with eight learned scalars shared over RGB."""

    def __init__(self):
        super().__init__()
        self.horizontal = LiftingAxis(-1)
        self.vertical = LiftingAxis(-2)

    def forward(self, image):
        if image.ndim != 4 or any(size % 2 for size in image.shape[-2:]):
            raise ValueError("Expected an NCHW tensor with even height and width.")
        low_x, high_x = self.horizontal(image)
        ll, hl = self.vertical(low_x)
        lh, hh = self.vertical(high_x)
        # Explicit convention: LH = differences along columns, HL = rows.
        return torch.stack((ll, lh, hl, hh), dim=2)

    def inverse(self, bands):
        if bands.ndim != 5 or bands.shape[2] != 4:
            raise ValueError("Expected B x C x 4 x H x W coefficients.")
        ll, lh, hl, hh = bands.unbind(dim=2)
        low_x = self.vertical.inverse(ll, hl)
        high_x = self.vertical.inverse(lh, hh)
        return self.horizontal.inverse(low_x, high_x)


class WaveletPool(nn.Module):
    def __init__(self):
        super().__init__();self.analysis=LearnableWavelet2D();self.downsample=nn.AvgPool2d(2,stride=2)

    def forward(self,image):
        bands=self.analysis(image.float())
        first=bands.permute(0,2,1,3,4).reshape(image.shape[0],12,*bands.shape[-2:])/2
        maps=[first]
        for _ in range(3):maps.append(self.downsample(maps[-1]))
        return maps
