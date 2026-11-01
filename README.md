# GazeLIFT

**Gaze-Guided Few-Shot Adaptation for Medical Image Segmentation**

Research manuscript for ISBI. This repository contains the project website, supplied paper, archived model implementation and reported experimental tables.

- [Project website](https://mohammedoussamaben.github.io/GazeLIFT/)
- [Paper](docs/assets/gazelift-paper.pdf)
- [Original model and training source](reproduction/seed42/)
- [Reported results and 15-run analysis](results/)

## Method

Frozen DINOv3 ViT-L/16 features at blocks 6, 12, 18 and 24 are conditioned on duration-weighted gaze and prototype contrast. A lifting-wavelet branch and compact decoder produce segmentation masks. The adapter has **817,993 trainable parameters**, excluding the frozen encoder. Gaze is required at inference.

## Source status

`reproduction/seed42/` preserves the paper's original implementation and import structure. It is an archived research pipeline, not a turnkey pretrained inference package. Its training/evaluation scripts expect prepared feature-cache manifests and local dataset paths. Read their argument parsers before running. No pretrained adapter checkpoint or medical dataset is included. Do not use random weights as a trained demonstration.

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

Serve `docs/` with any static HTTP server, for example `python -m http.server 8000 --directory docs`. GitHub Pages can publish this directory. `hf-space/` contains a static Space version of the research page; it does not perform model inference.

## Citation and rights

See [BibTeX](docs/citation.bib). Publication acceptance is not asserted. Third-party datasets, backbone weights and methods retain their own terms. No additional license for manuscript figures or pretrained weights is implied.
