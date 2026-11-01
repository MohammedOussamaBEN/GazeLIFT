"""Cached frozen DINO, original images, and new image-resolution expert gaze."""
from pathlib import Path
import csv,hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from image_data import load_image
from heatmap import make_image_heatmap


def load_torch(path):return torch.load(path,map_location='cpu',weights_only=False,mmap=True)


def fixation_records(manifest):
    path=Path(manifest['root'])/manifest['fixation_csv'];gaze={}
    with path.open(encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):gaze.setdefault(Path(row['IMAGE']).stem,[]).append(row)
    return gaze,hashlib.sha256(path.read_bytes()).hexdigest()


class FeatureImageDataset(Dataset):
    def __init__(self,cache,images,ids,manifest,sigma_px,augment=False):
        self.ids=ids;self.augment=augment
        self.rows=[load_torch(Path(cache)/(name+'.pt')) for name in ids]
        self.images=[load_image(images[name],512,row['native'].shape[-2:]) for name,row in zip(ids,self.rows)]
        gaze,self.fixation_csv_sha256=fixation_records(manifest);self.heatmaps=[];counts=[]
        for name in ids:
            records=sorted(gaze[name],key=lambda row:float(row['CURRENT_FIX_INDEX']))
            xy=np.asarray([[float(row['CURRENT_FIX_X']),float(row['CURRENT_FIX_Y'])] for row in records],np.float32)
            duration=np.asarray([float(row['CURRENT_FIX_DURATION']) for row in records],np.float32)
            assert len(records)>0 and np.isfinite(xy).all() and np.isfinite(duration).all()
            assert ((xy>=0)&(xy<=1)).all() and (duration>=0).all(),name
            # Same per-image duration transform as the trusted prior cache producer.
            duration=(duration-duration.min())/(duration.max()-duration.min()+1e-8) if duration.max()>duration.min() else np.ones_like(duration)
            h=make_image_heatmap(xy,duration,image_size=(512,512),sigma_px=sigma_px)
            assert h.shape==(1,512,512) and h.max()>0
            self.heatmaps.append(h);counts.append(len(records))
        self.gaze_audit={'n':len(ids),'sigma_px':sigma_px,'shape':[1,512,512],
                         'min_fixations':min(counts),'max_fixations':max(counts),'mean_fixations':float(np.mean(counts)),
                         'source':'original CSV coordinates, not upsampled cached heatmaps',
                         'duration_transform':'per-image min-max; constant durations -> all ones',
                         'fixation_csv_sha256':self.fixation_csv_sha256}

    def __len__(self):return len(self.ids)

    def __getitem__(self,index):
        source=self.rows[index]
        row={k:source[k] for k in ['id','features','mask','native']}
        row.update(image=self.images[index],gaze=self.heatmaps[index])
        if self.augment:
            dims=[]
            if torch.rand(())<.5:dims.append(-1)
            if torch.rand(())<.5:dims.append(-2)
            if dims:
                row['features']=[torch.flip(f,dims) for f in row['features']]
                for key in ['image','gaze','mask']:row[key]=torch.flip(row[key],dims)
        return row


def collate(rows):
    return {'features':[torch.stack([r['features'][i] for r in rows]) for i in range(4)],
            **{k:torch.stack([r[k] for r in rows]) for k in ['image','gaze','mask']},
            'id':[r['id'] for r in rows],'native':[r['native'] for r in rows]}
