from pathlib import Path
import csv,hashlib,json,shutil,tarfile
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from model import TransposeCNNGazePyramidDecoder
from data import FeatureImageDataset,collate,load_torch
from train import evaluate,summary,atomic_json

root=Path(__file__).resolve().parent.parent;report=root/'report';report.mkdir()
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=True
for name,digest in json.loads((root/'source_hashes.json').read_text()).items():assert hashlib.sha256((root/'code'/name).read_bytes()).hexdigest()==digest
selected={};checkpoints={}
for ds in ['kvasir','prostate']:
    folder=root/f'{ds}_seed42';settings=json.loads((folder/'settings.json').read_text())
    trained=json.loads((folder/'train_summary.json').read_text());history=json.loads((folder/'history.json').read_text())
    assert settings['dino_used'] and settings['dino_trainable_parameters']==0
    assert trained['best_epoch']>0 and trained['best_epoch'] in [r['epoch'] for r in history]
    assert abs(max(r['validation']['dice_mean'] for r in history)-trained['validation']['dice_mean'])<1e-7
    digest=hashlib.sha256((folder/'best.pt').read_bytes()).hexdigest();assert digest==trained['checkpoint_sha256']
    checkpoints[ds]={'path':str(folder/'best.pt'),'sha256':digest,'best_epoch':trained['best_epoch'],'validation':trained['validation']}
atomic_json(report/'final_selection.json',checkpoints)
comparison={};verified={}
for ds in ['kvasir','prostate']:
    folder=root/f'{ds}_seed42';settings=json.loads((folder/'settings.json').read_text())
    manifest=json.loads((root/'manifests'/f'{ds}_manifest.json').read_text())
    assert settings['manifest_sha256']==hashlib.sha256((root/'manifests'/f'{ds}_manifest.json').read_bytes()).hexdigest()
    model=TransposeCNNGazePyramidDecoder();state=load_torch(folder/'best.pt');model.load_state_dict(state['model']);model=model.cuda()
    assert sum(p.numel() for p in model.parameters())==settings['trainable_parameters']
    images={p.stem:p for p in (Path(manifest['root'])/'images').iterdir() if p.is_file()}
    dataset=FeatureImageDataset(Path(settings['existing_cache']),images,manifest['test'],manifest,settings['sigma_px'])
    loader=DataLoader(dataset,batch_size=4,num_workers=0,pin_memory=True,collate_fn=collate)
    masks=folder/'test_masks';masks.mkdir();records=evaluate(model,loader,masks)
    assert [r['id'] for r in records]==manifest['test']
    for r in records:
        p=np.asarray(Image.open(masks/(r['id']+'.png')))>0
        target=load_torch(Path(settings['existing_cache'])/(r['id']+'.pt'))['native'].numpy().squeeze()>.5
        intersection=(p&target).sum();dice=(2*intersection+1e-6)/(p.sum()+target.sum()+1e-6)
        iou=(intersection+1e-6)/((p|target).sum()+1e-6)
        np.testing.assert_allclose([r['dice'],r['iou']],[dice,iou],atol=1e-12,rtol=0)
    metrics={**summary(records),'dataset':ds,'seed':42,'trainable_parameters':settings['trainable_parameters'],
             'best_epoch':state['epoch'],'threshold':.5,'dino_used':True,'frozen_pretrained_dino':True,'segmentation_checkpoint_imported':False}
    atomic_json(folder/'test_summary.json',metrics);atomic_json(folder/'test_gaze_audit.json',dataset.gaze_audit)
    with (folder/'test_per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    dest=report/f'{ds}_seed42';dest.mkdir()
    for p in folder.iterdir():
        if p.suffix in ['.json','.csv']:shutil.copy2(p,dest/p.name)
    shutil.copytree(masks,dest/'test_masks')
    comparison[ds]=metrics;verified[ds]={'native_masks_verified':len(records),'checkpoint_sha256':checkpoints[ds]['sha256']}
    print(json.dumps(metrics),flush=True);del model,dataset,loader;torch.cuda.empty_cache()
atomic_json(report/'comparison.json',comparison);atomic_json(report/'verification.json',verified)
shutil.copytree(root/'code',report/'code',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(root/'manifests',report/'manifests')
for name in ['data_preflight.json','source_hashes.json']:shutil.copy2(root/name,report/name)
archive=root/'gaze_pyramid_transpose_cnn_decoder_rgb_seed42_report.tar.gz'
with tarfile.open(archive,'w:gz') as stream:stream.add(report,arcname='report')
archive.with_suffix('.gz.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
jobs=json.loads((root/'jobs.json').read_text());jobs.update(status='completed',test_metrics=comparison)
atomic_json(root/'jobs.json',jobs)
