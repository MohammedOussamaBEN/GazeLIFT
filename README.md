<h1 align="center">GazeLIFT</h1>
<p align="center"><strong>Gaze-Guided Few-Shot Adaptation for Medical Image Segmentation</strong></p>

<p align="center">
  <a href="https://mohammedoussamaben.github.io/GazeLIFT/"><img src="https://img.shields.io/badge/Website-285CE7?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project website"></a>
  <a href="https://huggingface.co/spaces/oussamab2n/GazeLIFT"><img src="https://img.shields.io/badge/Segmentation%20demo-FFB144?style=for-the-badge&logo=huggingface&logoColor=17243A" alt="Hugging Face segmentation demo"></a>
  <a href="https://huggingface.co/spaces/oussamab2n/GazeLIFT/tree/main/weights"><img src="https://img.shields.io/badge/Trained%20weights-168C86?style=for-the-badge&logo=huggingface&logoColor=white" alt="Trained adapter weights"></a>
</p>

<p align="center">
  <a href="#method">Method</a> · <a href="#results">Results</a> · <a href="#use-the-demo">Demo</a> · <a href="#use-the-code">Code</a> · <a href="#repository-contents">Repository</a>
</p>

GazeLIFT segments polyps and prostate MRI slices from an image and gaze fixations. It combines frozen DINOv3 features with duration-weighted gaze conditioning, learnable lifting wavelets and a compact decoder. The segmentation adapter has **817,993 trainable parameters**; the frozen encoder is excluded from this count and remains required at inference.

## Method

<p align="center">
  <img src="docs/assets/architecture.jpg" alt="GazeLIFT architecture: frozen DINOv3 features, gaze conditioning, lifting-wavelet image features and segmentation decoder" width="100%">
</p>

| Component | Implementation |
| --- | --- |
| Semantic features | Frozen DINOv3 ViT-L/16 outputs at blocks 6, 12, 18 and 24 |
| Gaze input | Fixation coordinates and durations, converted to a Gaussian heatmap |
| Conditioning | Foreground/background prototype contrast at each feature depth |
| Image features | Separable lifting-wavelet transform with 8 learned scalar parameters, followed by fixed average pooling |
| Decoder | Transposed convolutions, multi-scale feature fusion and an RGB skip |

The model uses 512×512 input images. Gaze records are required during both training and inference.

## Results

| Dataset | Main experiment: Dice (%) | 15 training runs: Dice (%) |
| --- | ---: | ---: |
| Kvasir-SEG | **93.35** | **93.22 ± 0.44** |
| NCI-ISBI | **89.24** | **87.94 ± 0.79** |

The last column reports mean ± SD across training runs on fixed test sets. These averages are separate from the main experiment. Protocols, confidence intervals and the training-fraction experiment are described on the [website](https://mohammedoussamaben.github.io/GazeLIFT/#results).

## Use the demo

Open the [Hugging Face Space](https://huggingface.co/spaces/oussamab2n/GazeLIFT).

1. Upload an image or a single grayscale DICOM slice.
2. Add fixation points by clicking the target, or import gaze records from CSV/XLSX/XLS.
3. Select the Kvasir-SEG or NCI-ISBI model and run segmentation.

The output tabs contain the segmentation overlay, binary mask and gaze prior. Images can be downloaded from each tab. The Space runs on ZeroGPU; requests may queue.

The [weights directory](https://huggingface.co/spaces/oussamab2n/GazeLIFT/tree/main/weights) contains both trained adapters as safetensors files and a manifest with their SHA-256 hashes. The interface source is available in the [Space repository](https://huggingface.co/spaces/oussamab2n/GazeLIFT/tree/main).

## Use the code

```bash
git clone https://github.com/MohammedOussamaBEN/GazeLIFT.git
cd GazeLIFT
python -m pip install -r requirements.txt
```

Instantiate the segmentation adapter:

```python
import sys
sys.path.insert(0, "model_code")

from model import TransposeCNNGazePyramidDecoder

model = TransposeCNNGazePyramidDecoder()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
assert trainable == 817_993
```

This initializes the architecture. Trained inference additionally requires the adapter weights, frozen DINOv3 backbone and gaze inputs. The [Space source](https://huggingface.co/spaces/oussamab2n/GazeLIFT/tree/main) contains the complete inference and weight-loading code.

<details>
<summary>Adapter inputs</summary>

| Input | Shape at 512×512 resolution |
| --- | --- |
| Normalized RGB image | `(B, 3, 512, 512)` |
| Frozen DINOv3 feature maps | Four tensors of shape `(B, 1024, 32, 32)` |
| Gaze heatmap | `(B, 1, 512, 512)` |
| Output logits | `(B, 1, 512, 512)` |

Apply sigmoid, resize probabilities to the native image resolution with bilinear interpolation, then threshold at `> 0.5` for the default binary mask.

</details>

## Repository contents

```text
model_code/         Architecture, data handling, training and evaluation source
docs/               Website source and figures
requirements.txt    Dependencies for the model and training code
```

Training requires CUDA, prepared DINOv3 feature caches, dataset manifests, original images, fixation records and masks. The scripts retain the original experiment layout and input checks. These data and caches are not included in the repository.

## Data and backbone access

Obtain the datasets and DINOv3 backbone under their respective access and licensing terms. Trained adapter weights are hosted in the Hugging Face Space linked above.
