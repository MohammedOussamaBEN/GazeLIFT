"""GazeLIFT trained-model Gradio interface."""
from pathlib import Path
import tempfile,json
import numpy as np
from PIL import ImageDraw
import gradio as gr
from inference import load_image,read_fixations,predict,validate_points

def draw(image,points):
    if image is None:return None
    v=image.copy();p=ImageDraw.Draw(v);r=max(3,min(image.size)//90)
    for i,(x,y,d) in enumerate(points):
        p.ellipse((x-r,y-r,x+r,y+r),fill='#ffe22b',outline='#8d7400');p.text((x+r+1,y-r),str(i+1),fill='#ffe22b')
    return v

def upload(path):
    if not path:return None,'',[],None,[],None,None,None,None,None,'Upload an image.'
    try:i=load_image(path)
    except Exception as e:raise gr.Error(str(e)) from e
    return i,Path(path).name,[],i,[],None,None,None,None,None,'Add fixations or import CSV.'

def add(image,points,duration,evt:gr.SelectData):
    if image is None:raise gr.Error('Upload an image first.')
    points=[list(p) for p in (points or [])]+[[float(evt.index[0]),float(evt.index[1]),float(duration)]]
    return points,draw(image,points),points

def import_csv(path,image,name,units):
    if image is None or not path:raise gr.Error('Upload an image and fixation CSV first.')
    try:p=read_fixations(path,name,image.size,units)
    except Exception as e:raise gr.Error(str(e)) from e
    return p,draw(image,p),p

def apply(image,rows):
    if image is None:raise gr.Error('Upload an image first.')
    try:p=validate_points(rows,image.size).tolist()
    except Exception as e:raise gr.Error(str(e)) from e
    return p,draw(image,p)

def segment(image,rows,dataset):
    if image is None:raise gr.Error('Upload an image first.')
    try:overlay,mask,probability,record=predict(image,rows,dataset)
    except Exception as e:raise gr.Error(str(e)) from e
    folder=Path(tempfile.mkdtemp(prefix='gazelift-'));mask.save(folder/'mask.png');np.save(folder/'probability.npy',probability)
    info={'dataset':dataset,'checkpoint_sha256':record['sha256'],'sigma_px':record['sigma_px'],'threshold':.5,'native_size':list(image.size),'fixations':len(rows)}
    (folder/'settings.json').write_text(json.dumps(info,indent=2))
    return overlay,mask,str(folder/'mask.png'),str(folder/'probability.npy'),str(folder/'settings.json'),'Completed with verified trained weights.'

with gr.Blocks(title='GazeLIFT') as demo:
    gr.Markdown('# GazeLIFT\n### Gaze-guided medical image segmentation\nUpload an image, provide fixation points, and choose the trained adapter.')
    image=gr.State(None);name=gr.State('');points=gr.State([])
    with gr.Row():
        with gr.Column():
            file=gr.File(label='Image or single grayscale DICOM slice',file_types=['image','.dcm'],type='filepath')
            dataset=gr.Dropdown(['kvasir','prostate'],value='kvasir',label='Trained adapter: Kvasir-SEG / NCI-ISBI')
            canvas=gr.Image(label='Click to add fixations',type='pil',interactive=False)
            duration=gr.Number(value=1,minimum=0,label='Duration for next fixation')
            clear=gr.Button('Clear fixations')
        with gr.Column():
            csv_file=gr.File(label='Fixation CSV',file_types=['.csv'],type='filepath')
            units=gr.Radio(['Normalized [0,1]','Native pixels'],value='Normalized [0,1]',label='CSV coordinate units')
            import_button=gr.Button('Import fixation CSV')
            table=gr.Dataframe(headers=['x','y','duration'],datatype=['number']*3,type='array',value=[],column_count=(3,'fixed'),label='Fixations in native pixels',interactive=True)
            apply_button=gr.Button('Apply table edits')
            run=gr.Button('Segment with trained model',variant='primary')
            status=gr.Markdown('Requires installed trained adapters. CPU inference may take time.')
    with gr.Row():
        overlay=gr.Image(label='Prediction overlay',type='pil');mask=gr.Image(label='Binary segmentation',type='pil')
    with gr.Row():
        mask_file=gr.File(label='Mask PNG');prob_file=gr.File(label='Probability array');settings=gr.File(label='Inference settings')
    gr.Markdown('CSV: `x,y,duration` or `CURRENT_FIX_X,CURRENT_FIX_Y,CURRENT_FIX_DURATION`. Duration is optional; an optional `IMAGE` / `image_name` column matches the uploaded filename. The editable table supplies the segmentation input. No randomly initialized fallback is used.')
    file.change(upload,[file],[image,name,points,canvas,table,overlay,mask,mask_file,prob_file,settings,status])
    canvas.select(add,[image,points,duration],[points,canvas,table])
    clear.click(lambda i:([],draw(i,[]),[]),[image],[points,canvas,table])
    import_button.click(import_csv,[csv_file,image,name,units],[points,canvas,table])
    apply_button.click(apply,[image,table],[points,canvas])
    run.click(segment,[image,table,dataset],[overlay,mask,mask_file,prob_file,settings,status],concurrency_limit=1)
if __name__=='__main__':demo.queue(default_concurrency_limit=1).launch()
