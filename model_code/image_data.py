"""Match the trusted encoder-cache producer's raw image preprocessing exactly."""
import numpy as np
from PIL import Image
import torch

MEAN=torch.tensor([.485,.456,.406])[:,None,None]
STD=torch.tensor([.229,.224,.225])[:,None,None]

def load_image(path,resolution,expected_native):
    if path.suffix.lower()=='.dcm':
        import pydicom
        arr=pydicom.dcmread(path).pixel_array.astype(np.float32)
        arr=(arr-arr.min())/max(float(arr.max()-arr.min()),1e-8)
        image=Image.fromarray((arr*255).astype(np.uint8)).convert('RGB')
    else:image=Image.open(path).convert('RGB')
    assert image.size==(int(expected_native[-1]),int(expected_native[-2])), 'Image/native-mask shape mismatch'
    image=image.resize((resolution,resolution),Image.Resampling.BILINEAR)
    return ((torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255-MEAN)/STD).half()
