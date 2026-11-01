---
title: GazeLIFT
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# GazeLIFT interactive segmentation

Upload an RGB image or a single grayscale DICOM slice, add fixation points by clicking or import a fixation CSV, select the trained dataset adapter, and download native-resolution segmentation outputs.

## Setup

```sh
pip install -r requirements.txt
python app.py
```

Install the original validation-selected adapters as `weights/kvasir_seed42.pt` and `weights/prostate_seed42.pt`. Alternatively set `GAZELIFT_WEIGHTS_REPO` to a Hugging Face model repository containing these filenames; the app downloads them at runtime. Set the `HF_TOKEN` Space secret only if the repository requires authentication. Local overrides are `GAZELIFT_KVASIR_CHECKPOINT` and `GAZELIFT_PROSTATE_CHECKPOINT`.

The manifest records the original checkpoint SHA-256 hashes. Loading stops if a checkpoint is missing, mismatched or incompatible. No trained adapter files are currently included. The weights must be obtained from the completed training runs before deploying a functional segmentation demo.

The frozen encoder is `vit_large_patch16_dinov3.lvd1689m`, loaded separately through timm subject to the backbone's access and licensing terms. Set `GAZELIFT_DINO_CHECKPOINT` for an independently obtained local checkpoint. The app uses CUDA when available; CPU inference requires sufficient RAM and may be slow.

## Inputs and outputs

- `kvasir`: trained polyp adapter; gaze bandwidth 32 pixels at 512×512.
- `prostate`: trained NCI-ISBI adapter; gaze bandwidth 8 pixels at 512×512.
- Clicks and the editable table use native pixel coordinates. Durations default to 1; change the value before each click.
- CSV columns: `x,y,duration` or `CURRENT_FIX_X,CURRENT_FIX_Y,CURRENT_FIX_DURATION`. Duration is optional. Select normalized or native-pixel units explicitly. An optional `IMAGE` / `image_name` column matches rows to the uploaded image filename.
- The editable table supplies the segmentation input. **Apply table edits** refreshes the drawn markers.
- Downloads include a binary mask PNG, floating-point probability array and inference settings JSON.

Preprocessing follows the archived implementation: 512×512 ImageNet normalization, raw DINO outputs at blocks 6/12/18/24, fp16 cache quantization followed by float32 adapter inference, per-image min-max duration weights, and native-resolution bilinear probability interpolation before thresholding at >0.5. Equal durations produce uniform fixation weights.

## Validation and deployment

Input parsing, heatmap construction, rejection of missing checkpoints and architecture forward execution are tested. End-to-end predictions must also be compared against saved native reference masks when the matching trained checkpoints become available. Numerical reproduction is not yet verified.

Deploy this directory as a Gradio Space under `oussamab2n/GazeLIFT` after signing in with write access. Publish the verified adapters in a separate model repository and configure `GAZELIFT_WEIGHTS_REPO`, or add them directly to the Space. Do not redistribute datasets or third-party backbone weights without permission. Requests are queued one at a time.
