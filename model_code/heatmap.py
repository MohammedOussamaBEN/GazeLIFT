"""Render duration-weighted Gaussian gaze directly on the input image grid."""
import torch


def make_image_heatmap(xy,durations,image_size=(512,512),sigma_px=16.):
    """xy: Nx2 normalized coordinates (x,y); durations: N nonnegative weights.

    Sigma is in pixels on the resized model-input image, not DINO patch units.
    The duration-weighted Gaussian sum is normalized by its maximum to [0,1].
    No eye-tracker coordinates are inferred from a coarse cached heatmap.
    """
    xy=torch.as_tensor(xy,dtype=torch.float32).reshape(-1,2)
    durations=torch.as_tensor(durations,dtype=torch.float32,device=xy.device).reshape(-1)
    height,width=image_size
    if len(xy)!=len(durations) or height<=0 or width<=0 or sigma_px<=0:
        raise ValueError('Invalid fixation lengths, image dimensions or Gaussian sigma.')
    if not torch.isfinite(xy).all() or not torch.isfinite(durations).all():
        raise ValueError('Fixation coordinates and durations must be finite.')
    if (xy<0).any() or (xy>1).any() or (durations<0).any():
        raise ValueError('Coordinates must be normalized to [0,1]; durations must be nonnegative.')
    # A separable Gaussian evaluates the same weighted sum without a costly
    # fixation-by-image stack. This also handles an empty fixation sequence.
    xx=torch.arange(width,device=xy.device)[None,:]
    yy=torch.arange(height,device=xy.device)[None,:]
    gx=torch.exp(-(xx-xy[:,0:1]*(width-1)).square()/(2*sigma_px**2))
    gy=torch.exp(-(yy-xy[:,1:2]*(height-1)).square()/(2*sigma_px**2))
    heatmap=(gy.transpose(0,1)*durations[None,:])@gx
    return (heatmap/heatmap.max().clamp_min(1e-12)).unsqueeze(0)
