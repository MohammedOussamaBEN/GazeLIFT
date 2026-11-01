"""Dataset-specific GazeLIFT inference; no random-weight fallback."""
from pathlib import Path
from functools import lru_cache
import csv,hashlib,json,os,sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'model_code'))
from model import TransposeCNNGazePyramidDecoder
from heatmap import make_image_heatmap

def load_image(path):
    if Path(path).suffix.lower()=='.dcm':
        import pydicom
        a=pydicom.dcmread(path).pixel_array.astype(np.float32)
        if a.ndim!=2 or not np.isfinite(a).all():raise ValueError('Upload one finite grayscale DICOM slice.')
        a=(a-a.min())/max(float(a.max()-a.min()),1e-8)
        return Image.fromarray((a*255).astype(np.uint8)).convert('RGB')
    with Image.open(path) as i:return i.convert('RGB')

def validate_points(rows,size):
    a=np.asarray(rows,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]!=3 or len(a)==0 or not np.isfinite(a).all():raise ValueError('Provide finite x, y, duration values for at least one fixation.')
    if (a<0).any() or (a[:,0]>size[0]-1).any() or (a[:,1]>size[1]-1).any():raise ValueError('Fixations must be inside the image, with nonnegative durations.')
    return a

def prepare_gaze(rows,size,sigma):
    a=validate_points(rows,size);xy=a[:,:2]/np.array([max(size[0]-1,1),max(size[1]-1,1)])
    d=a[:,2];d=(d-d.min())/(d.max()-d.min()+1e-8) if d.max()>d.min() else np.ones_like(d)
    h=make_image_heatmap(xy,d,image_size=(512,512),sigma_px=sigma)
    if not torch.isfinite(h).all() or h.max()<=0:raise ValueError('Invalid gaze heatmap.')
    return h[None]

def read_fixations(path,name,size,units):
    with open(path,encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);keys={k.lower():k for k in reader.fieldnames or []}
        def col(*names):return next((keys[n] for n in names if n in keys),None)
        x,y=col('x','current_fix_x'),col('y','current_fix_y');d=col('duration','current_fix_duration');im=col('image','image_name');order=col('current_fix_index')
        if not x or not y:raise ValueError('CSV needs x/y or CURRENT_FIX_X/CURRENT_FIX_Y columns.')
        rows=[r for r in reader if not im or Path(r[im]).stem==Path(name).stem]
        if order:rows.sort(key=lambda r:float(r[order]))
        a=np.array([[float(r[x]),float(r[y]),float(r[d]) if d else 1.] for r in rows],np.float32)
    if not len(a):raise ValueError('No rows match the uploaded image name.')
    if units=='Normalized [0,1]':
        if ((a[:,:2]<0)|(a[:,:2]>1)).any():raise ValueError('Normalized coordinates must lie in [0,1].')
        a[:,:2]*=np.array([max(size[0]-1,1),max(size[1]-1,1)])
    return validate_points(a,size).tolist()

def verified_checkpoint(dataset):
    manifest=json.loads((ROOT/'weights/manifest.json').read_text());record=manifest[dataset]
    path=Path(os.environ.get(record['environment_variable'],str(ROOT/'weights'/record['filename'])))
    if not path.is_file():
        repo=os.environ.get('GAZELIFT_WEIGHTS_REPO')
        if not repo:raise FileNotFoundError('The verified trained adapter is not installed. Configure GAZELIFT_WEIGHTS_REPO or add the checkpoint file.')
        from huggingface_hub import hf_hub_download
        path=Path(hf_hub_download(repo_id=repo,filename=record['filename'],token=os.environ.get('HF_TOKEN')))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Checkpoint hash does not match the recorded trained adapter.')
    return path,record

@lru_cache(maxsize=2)
def load_adapter(dataset,device):
    path,record=verified_checkpoint(dataset)
    payload=torch.load(path,map_location='cpu',weights_only=True)
    model=TransposeCNNGazePyramidDecoder();model.load_state_dict(payload['model'],strict=True)
    return model.requires_grad_(False).eval().to(device),record

@lru_cache(maxsize=1)
def load_encoder(device):
    import timm
    local=os.environ.get('GAZELIFT_DINO_CHECKPOINT')
    model=timm.create_model('vit_large_patch16_dinov3.lvd1689m',pretrained=not bool(local),num_classes=0,img_size=512,dynamic_img_size=True)
    if local:timm.models.load_checkpoint(model,local,strict=True)
    return model.requires_grad_(False).eval().to(device)

@torch.inference_mode()
def predict(image,rows,dataset):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    adapter,record=load_adapter(dataset,device)
    gaze=prepare_gaze(rows,image.size,record['sigma_px']).to(device)
    small=image.convert('RGB').resize((512,512),Image.Resampling.BILINEAR)
    x=torch.from_numpy(np.asarray(small).copy()).permute(2,0,1).float()/255
    x=(x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    encoder=load_encoder(device);capture={};hooks=[]
    for i in [5,11,17,23]:hooks.append(encoder.blocks[i].register_forward_hook(lambda m,args,out,i=i:capture.update({i:out[:,encoder.num_prefix_tokens:]})))
    try:encoder(x[None].to(device))
    finally:
        for hook in hooks:hook.remove()
    features=[]
    for i in [5,11,17,23]:
        if capture[i].shape[1:]!=(1024,1024):raise ValueError('Unexpected frozen feature shape.')
        features.append(capture[i].transpose(1,2).reshape(1,1024,32,32).half().float())
    logits=adapter(x[None].half().float().to(device),features,gaze)
    probability=F.interpolate(logits.sigmoid(),size=(image.height,image.width),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
    mask=probability>.5;overlay=np.asarray(image.convert('RGB')).copy()
    overlay[mask]=(.55*overlay[mask]+.45*np.array([0,205,235])).astype(np.uint8)
    return Image.fromarray(overlay),Image.fromarray((255*mask).astype(np.uint8)),probability,record
