"""Fresh training of the one shared fuse-then-gaze architecture. Validation only."""
from pathlib import Path
import argparse,csv,hashlib,json,math,random,time
import copy
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import TransposeCNNGazePyramidDecoder,segmentation_loss
from data import FeatureImageDataset,collate,load_torch


def emit(x):print(json.dumps(x),flush=True)
def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def patient(name):return name.rsplit('_',1)[0]
def fingerprint(model):
    digest=hashlib.sha256()
    for name,p in model.state_dict().items():digest.update(name.encode());digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
def inputs(batch):
    return (batch['image'].cuda(non_blocking=True).float(),
            [f.cuda(non_blocking=True) for f in batch['features']],batch['gaze'].cuda(non_blocking=True).float())
def dice_iou(probability,target):
    p=probability>.5;t=target>.5;inter=(p&t).sum().item()
    return (2*inter+1e-6)/(p.sum().item()+t.sum().item()+1e-6),(inter+1e-6)/((p|t).sum().item()+1e-6)


@torch.no_grad()
def evaluate(model,loader,masks=None):
    model.eval();records=[]
    for batch in loader:
        probabilities=model(*inputs(batch)).sigmoid().float().cpu()
        for i,name in enumerate(batch['id']):
            native=batch['native'][i]
            p=F.interpolate(probabilities[i:i+1],size=native.shape[-2:],mode='bilinear',align_corners=False)[0]
            dice,iou=dice_iou(p,native)
            records.append({'id':name,'patient':patient(name),'dice':dice,'iou':iou})
            if masks is not None:Image.fromarray(((p[0]>.5).numpy()*255).astype(np.uint8)).save(masks/(name+'.png'))
    return records


def summary(records):
    result={'n':len(records)}
    for key in ['dice','iou']:
        values=np.array([r[key] for r in records])*100
        result[key+'_mean']=float(values.mean());result[key+'_case_sd']=float(values.std(ddof=1))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=['kvasir','prostate'],required=True)
    ap.add_argument('--manifest',required=True);ap.add_argument('--existing-cache',required=True)
    ap.add_argument('--sigma-px',type=float,required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--epochs',type=int,default=80)
    ap.add_argument('--patience',type=int,default=15);ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--accumulation',type=int,default=2);ap.add_argument('--lr',type=float,default=2e-4)
    args=ap.parse_args();assert args.seed==42 and torch.cuda.is_available()
    torch.set_num_threads(4);random.seed(args.seed);np.random.seed(args.seed)
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=True
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);assert not (root/'settings.json').exists()
    manifest=json.loads(Path(args.manifest).read_text());assert manifest['dataset']==args.dataset
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(manifest[a])&set(manifest[b])
        if args.dataset=='prostate':assert not {patient(x) for x in manifest[a]}&{patient(x) for x in manifest[b]}
    cache=Path(args.existing_cache);ids=manifest['train']+manifest['validation']+manifest['test']
    assert args.sigma_px==(32. if args.dataset=='kvasir' else 8.)
    meta=json.loads((cache/'cache_metadata.json').read_text())
    assert meta['dataset']==args.dataset and meta['resolution']==512 and meta['blocks']==[6,12,18,24]
    assert meta['ids_sha256']==hashlib.sha256('\n'.join(ids).encode()).hexdigest()
    assert meta['training_code_sha256']=='661fe6a153c3519229f6e8dcb3fb500dff5737a090faff86998f861a80df6b3d'
    atomic_json(root/'cache_metadata.json',meta)

    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py') if p.name in ['train.py','model.py','data.py','image_data.py','heatmap.py','base_model.py','wavelet.py','reference_model.py','width_reference.py','flat_reference.py','pointwise_reference.py','pyramid_reference.py','conv3_reference.py','selected_reference.py','transpose_reference.py']}
    model=TransposeCNNGazePyramidDecoder();params=sum(p.numel() for p in model.parameters())
    initial_parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
    settings={**vars(args),'architecture':'gaze_pyramid_transpose_cnn_decoder_rgb','resolution':512,'counts':manifest['counts'],
              'initialization':'all learned decoder/projection/conditioning weights newly random; wavelet Haar-initialized; only frozen pretrained DINO features retained',
              'segmentation_checkpoint_loaded':False,'pretrained_encoder':'frozen DINOv3 ViT-L/16 cached features','dino_used':True,'wavelet_used':True,'dino_trainable_parameters':0,
              'wavelet_channels':[12,12,12,12],'dino_blocks':[6,12,18,24],'dino_projection_channels':[32,64,128,256],'gaze_conditioned_widths':[32,64,128,256],
              'decoder_widths':[32,32,32,32,32],'bottleneck_width':32,'decoder_input_channels':[268,172,108,76,35],'deep_fusion_input_channels':268,'full_resolution_skip':'original normalized RGB3 concatenated with upsampled decoder32','decoder_block':'one ConvTranspose2d3x3 per block:268/172/108/76/35->32; strides2/2/2/2/1,padding1,output_padding1/1/1/1/0,biasFalse,GN8,GELU; then Conv2d32->32 kernel3,padding1,biasFalse,GN8,GELU','skip_merge':'concatenate previous transposed-decoder32 at the current gaze/wavelet grid; fuse before next learned upsampling',
              'gaze_duration_weighted':True,'heatmap_from_original_fixations':True,'sigma_units':'input pixels',
              'background_subtraction':True,'prototype_features':'projected DINO32/64/128/256 ONLY, resized to each scale; never wavelet-fused',
              'conditioning':'concat projected DINO,H1,C1; Conv1x1 inputs34/66/130/258, outputs32/64/128/256, GN8/GELU',
              'encoder_recurrence':'DINO-only conditioning independently per depth; wavelet branch bypasses gaze and enters only deep decoder fusion and three decoder concatenations',
              'pooling':'one learned8-coefficient lifting wavelet then three fixed average pools','upsampling':'decoder learned ConvTranspose2d; only DINO projection resizing remains fixed bilinear',
              'knn':False,'prototype_iterations':0,'tta':False,'threshold':.5,
              'loss':'BCEWithLogits + soft Dice, final head only','optimizer':'AdamW, weight_decay1e-4',
              'schedule':'5 epoch linear warmup then cosine to 0.05x LR',
              'augmentation':'paired H/V image, cached feature, full-resolution heatmap and mask flips; frozen DINO not re-encoded',
              'data_rng_seed':args.seed+1000,'effective_batch_size':args.batch_size*args.accumulation,
              'checkpoint_selection':'highest native validation mean Dice; no test use','test_evaluated':False,
              'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
              'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              'sources_sha256':sources,'initial_model_fingerprint':fingerprint(model),'same_shaped_retained_modules_start_as_baseline':True,'trainable_parameters':params,
              'component_parameters':{n:sum(p.numel() for p in getattr(model,n).parameters()) for n in ['wavelet','dino_projection','gaze_conditioning','decoder','mask_head']}}
    atomic_json(root/'settings.json',settings);emit({'phase':'random_initialization',**settings})
    model=model.cuda()
    images={p.stem:p for p in (Path(manifest['root'])/'images').iterdir() if p.is_file()};assert set(ids)<=set(images)
    def loader(split,augment=False):
        dataset=FeatureImageDataset(cache,images,manifest[split],manifest,args.sigma_px,augment)
        atomic_json(root/f'{split}_gaze_audit.json',dataset.gaze_audit)
        return DataLoader(dataset,batch_size=args.batch_size,shuffle=augment,num_workers=0,pin_memory=True,collate_fn=collate)
    train_loader=loader('train',True);val_loader=loader('validation')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    def multiplier(epoch):
        if epoch<5:return (epoch+1)/5
        progress=min(1.,(epoch-5)/max(args.epochs-5,1))
        return .05+.95*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    torch.manual_seed(args.seed+1000);torch.cuda.manual_seed_all(args.seed+1000)
    initial=summary(evaluate(model,val_loader));atomic_json(root/'initial_validation.json',initial);emit({'phase':'initial_validation','metrics':initial})
    best=initial['dice_mean'];best_epoch=0;stale=0;history=[];start=time.monotonic()
    torch.save({'model':model.state_dict(),'epoch':0,'validation':initial,'settings':settings},root/'best.pt')
    for epoch in range(1,args.epochs+1):
        model.train();total_loss=0.;seen=0;optimizer.zero_grad(set_to_none=True)
        for index,batch in enumerate(train_loader):
            logits=model(*inputs(batch));target=batch['mask'].cuda(non_blocking=True).float()
            loss=segmentation_loss(logits,target);assert torch.isfinite(loss)
            group_start=(index//args.accumulation)*args.accumulation
            examples=min(args.batch_size*args.accumulation,len(train_loader.dataset)-group_start*args.batch_size)
            (loss*(len(batch['id'])/examples)).backward()
            if (index+1)%args.accumulation==0 or index+1==len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
            total_loss+=loss.item()*len(batch['id']);seen+=len(batch['id'])
        val=summary(evaluate(model,val_loader));lr=optimizer.param_groups[0]['lr'];scheduler.step()
        record={'epoch':epoch,'train_loss':total_loss/seen,'lr':lr,'validation':val,'seconds':round(time.monotonic()-start,1)}
        history.append(record);atomic_json(root/'history.json',history);emit({'phase':'epoch',**record})
        if val['dice_mean']>best+1e-5:
            best=val['dice_mean'];best_epoch=epoch;stale=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'validation':val,'settings':settings},root/'best.pt')
        else:stale+=1
        if stale>=args.patience:break
    selected=load_torch(root/'best.pt');model.load_state_dict(selected['model'])
    records=evaluate(model,val_loader);chosen=summary(records);assert abs(chosen['dice_mean']-best)<1e-7
    with (root/'validation_per_image.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    for name,digest in sources.items():assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest
    changes={n:float((p.detach().cpu()-initial_parameters[n]).norm()) for n,p in model.named_parameters()}
    assert all(v>1e-10 and np.isfinite(v) for v in changes.values())
    atomic_json(root/'parameter_updates.json',changes)
    result={'dataset':args.dataset,'seed':args.seed,'best_epoch':best_epoch,'epochs_completed':len(history),
            'validation':chosen,'training_seconds':time.monotonic()-start,'test_evaluated':False,
            'initial_model_fingerprint':settings['initial_model_fingerprint'],
            'checkpoint_sha256':hashlib.sha256((root/'best.pt').read_bytes()).hexdigest()}
    atomic_json(root/'train_summary.json',result);emit({'phase':'training_complete',**result})

if __name__=='__main__':main()
