# GazeLIFT

**Gaze-Guided Few-Shot Adaptation for Medical Image Segmentation**

Gaze-guided medical image segmentation with a frozen DINOv3 encoder and a compact trainable adapter. This repository contains the project website, model implementation, interactive demo source and experimental tables.

- [Project website](https://mohammedoussamaben.github.io/GazeLIFT/)
- [Original model and training source](reproduction/seed42/)
- [Reported results and 15-run analysis](results/)

## Method

Frozen DINOv3 ViT-L/16 features at blocks 6, 12, 18 and 24 are conditioned on duration-weighted gaze and prototype contrast. A lifting-wavelet branch and compact decoder produce segmentation masks. The adapter has **817,993 trainable parameters**, excluding the frozen encoder. Gaze is required at inference.

## Source status

`reproduction/seed42/` preserves the original implementation and import structure. It is an archived research pipeline, not a turnkey pretrained inference package. Its training/evaluation scripts expect prepared feature-cache manifests and local dataset paths. Read their argument parsers before running. No pretrained adapter checkpoint or medical dataset is included. Do not use random weights as a trained demonstration.

For architecture inspection, install PyTorch and import from the archived directory:

```python
import sys
sys.path.insert(0, "reproduction/seed42")
from model import TransposeCNNGazePyramidDecoder
model = TransposeCNNGazePyramidDecoder()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
# 817993
```

The training pipeline also requires NumPy, Pillow and pydicom for DICOM inputs. DINOv3 feature extraction requires the separately obtained backbone and its applicable license/access terms. Dataset reuse and gaze permissions must be obtained independently.

## Results

Main seed-42 Dice: **93.35%** on Kvasir-SEG and **89.24%** on NCI-ISBI. The separate fixed-update training-fraction experiment has different reference results and must not be merged with the main comparison.

Across seeds 40–54, mean ± SD Dice is **93.22 ± 0.44%** and **87.94 ± 0.79%** with gaze, compared with **90.28 ± 0.54%** and **84.25 ± 0.88%** without gaze. Confidence intervals and paired analysis are in `results/`; SD describes training-run variation on fixed test sets. Original source tables retain their original numerical precision.

## Website

Serve `docs/` with any static HTTP server, for example `python -m http.server 8000 --directory docs`. GitHub Pages can publish this directory. `hf-space/` contains the Gradio inference application. It requires the dataset-specific verified trained adapters and the separately obtained DINOv3 backbone. See its README for setup.

## Trained-model demo

The Gradio application accepts RGB images or individual grayscale DICOM slices, click-based fixations or a CSV fixation file, and returns native-resolution probability maps, masks and overlays. It selects the Kvasir-SEG or NCI-ISBI adapter explicitly. There is no randomly initialized inference fallback.

Adapter checkpoint availability and Hugging Face account access are pending. See `hf-space/weights/manifest.json` for the recorded SHA-256 digests of the original selected adapters. The demo must pass a native-mask comparison against saved reference predictions before it is described as reproducing reported results.

## Rights

Third-party datasets and backbone weights retain their own access and licensing terms. No datasets or backbone weights are redistributed here.
