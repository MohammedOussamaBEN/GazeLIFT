"""Verify original trained files, export clean tensors, and publish model + Space."""
import argparse,hashlib,json,sys,tempfile
from pathlib import Path
import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'hf-space'))
from inference import TransposeCNNGazePyramidDecoder

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kvasir',type=Path,required=True)
    p.add_argument('--prostate',type=Path,required=True)
    p.add_argument('--publish',action='store_true',help='Upload only after successful local verification.')
    args=p.parse_args()
    manifest_path=ROOT/'hf-space/weights/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    with tempfile.TemporaryDirectory(prefix='gazelift-weights-') as folder:
        dest=Path(folder)
        for name,path in [('kvasir',args.kvasir),('prostate',args.prostate)]:
            record=manifest[name]
            expected=record.get('original_checkpoint_sha256',record['sha256'])
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            if actual!=expected:raise ValueError(f'{name}: original checkpoint hash mismatch')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            state=payload['model']
            model=TransposeCNNGazePyramidDecoder();model.load_state_dict(state,strict=True)
            clean={k:v.detach().cpu().contiguous() for k,v in state.items()}
            filename=f'{name}_seed42.safetensors'
            save_file(clean,str(dest/filename))
            record.update(original_checkpoint_sha256=actual,filename=filename,
                          sha256=hashlib.sha256((dest/filename).read_bytes()).hexdigest())
        (dest/'README.md').write_text('''---
library_name: pytorch
tags:
- image-segmentation
- medical-imaging
---
# GazeLIFT trained adapters

Dataset-specific Kvasir-SEG and NCI-ISBI adapters with 817,993 trainable parameters. Requires the separately obtained frozen DINOv3 ViT-L/16 backbone and gaze fixation records. These files contain model tensors only; datasets, backbone weights and training-path metadata are excluded.

Use with the GazeLIFT segmentation Space. Validation selects seed-42 checkpoint epochs 23 and 33, respectively. The configuration and checkpoint digests are provided in the manifest. The clean tensor export does not change model weights. Native-reference inference comparison remains a separate required verification step.
''',encoding='utf-8')
        (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if not args.publish:
            out=ROOT/'output/verified_weights';out.mkdir(parents=True,exist_ok=True)
            import shutil
            for f in dest.iterdir():shutil.copy2(f,out/f.name)
            print('Verified clean adapter exports are in output/verified_weights. No upload performed.')
            return
        api=HfApi();who=api.whoami()
        if who.get('name')!='oussamab2n':raise PermissionError('Sign in as oussamab2n before publishing.')
        api.create_repo('oussamab2n/GazeLIFT',repo_type='model',private=False,exist_ok=True)
        api.upload_folder(repo_id='oussamab2n/GazeLIFT',repo_type='model',folder_path=str(dest),commit_message='Release verified trained adapter tensors')
        manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
        api.create_repo('oussamab2n/GazeLIFT',repo_type='space',space_sdk='gradio',private=False,exist_ok=True)
        api.upload_folder(repo_id='oussamab2n/GazeLIFT',repo_type='space',folder_path=str(ROOT/'hf-space'),ignore_patterns=['__pycache__/**','*.pyc'],commit_message='Deploy trained-model segmentation interface')
        api.add_space_variable('oussamab2n/GazeLIFT','GAZELIFT_WEIGHTS_REPO','oussamab2n/GazeLIFT')
        print('Published model repository and segmentation Space under oussamab2n/GazeLIFT.')

if __name__=='__main__':main()
