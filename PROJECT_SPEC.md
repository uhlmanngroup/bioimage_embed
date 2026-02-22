# bioimage_embed - Project Specification

> Thin, maintainable wrapper for biological image embeddings using modern ML infrastructure.

## Problem Statement

Natural image models (ImageNet-pretrained) don't work well with biological images:
- Different statistics (fluorescence, phase contrast, etc.)
- Multi-channel (not RGB)
- Scale/resolution matters (subcellular vs tissue)
- Biologists need interpretable features, not just predictions

## Vision

A **minimal, opinionated CLI** that makes it trivial to:
1. Extract embeddings from microscopy images
2. Fine-tune foundation models on biological data
3. Produce standard Python science objects (DataFrames, AnnData, zarr)

## Design Principles

1. **Thin wrapper** — delegate to HuggingFace, timm, not reinvent
2. **Minimal config** — Hydra-zen, Python-defined schemas, almost no YAML
3. **CLI-first** — all features accessible via command line
4. **Standard outputs** — pandas DataFrame, AnnData, zarr arrays
5. **Dask for scale** — tiling, sliding windows, lazy evaluation
6. **Lightly maintainable** — <2000 LOC core, well-documented

## Core Features

### 1. Featurization (`bioimage-embed featurize`)
```bash
bioimage-embed featurize \
  --input images/*.tif \
  --model dinov2-vit-base \
  --output features.parquet \
  --tile-size 256 \
  --stride 128 \
  --batch-size 32
```

- **Tiling**: Dask-based sliding window for large images
- **Multi-channel**: Handle arbitrary channel counts
- **Output formats**: parquet, zarr, h5ad (AnnData)

### 2. Pretraining (`bioimage-embed pretrain`)
```bash
bioimage-embed pretrain \
  --data cellpaint/*.zarr \
  --method simclr \
  --backbone resnet50 \
  --output model.pt
```

- **Methods**: SimCLR, DINO, MAE, BarlowTwins
- **Backbones**: ResNet, ViT, ConvNeXt (via timm)

### 3. Fine-tuning (`bioimage-embed finetune`)
```bash
bioimage-embed finetune \
  --data train.csv \
  --model pretrained.pt \
  --task classification \
  --output finetuned.pt
```

- **Tasks**: classification, regression, segmentation
- **Few-shot**: support for small labeled datasets
- **Transfer**: from natural or bio-pretrained models

### 4. Analysis (`bioimage-embed analyze`)
```bash
bioimage-embed analyze \
  --features features.parquet \
  --method umap \
  --output analysis.h5ad
```

- **Dimensionality reduction**: PCA, UMAP, t-SNE
- **Clustering**: Leiden, k-means
- **Bioinformatics-style**: differential expression analogues

## Architecture

```
bioimage_embed/
├── cli.py              # Typer CLI, minimal
├── config.py           # Hydra-zen schemas (Python, not YAML)
├── featurize/
│   ├── models.py       # HuggingFace/timm wrapper
│   ├── tiling.py       # Dask sliding window
│   └── outputs.py      # DataFrame/AnnData/zarr writers
├── train/
│   ├── ssl.py          # Self-supervised (SimCLR, DINO, etc.)
│   ├── supervised.py   # Classification, regression
│   └── fewshot.py      # Few-shot learning
├── analyze/
│   ├── embeddings.py   # UMAP, clustering
│   └── cellpaint.py    # CellPaint-specific analysis
└── io/
    ├── readers.py      # bioformats, tifffile, zarr
    └── writers.py      # parquet, h5ad, zarr
```

## Config Philosophy (Hydra-zen)

**No logic in YAML**. Config is Python with type hints:

```python
# config.py
from dataclasses import dataclass
from hydra_zen import builds, make_config

@dataclass
class FeaturizeConfig:
    model: str = "dinov2-vit-base"
    tile_size: int = 256
    stride: int = 128
    batch_size: int = 32
    output_format: str = "parquet"  # parquet | zarr | h5ad

# Compose configs
Config = make_config(
    featurize=FeaturizeConfig,
    # ...
)
```

YAML only for overrides:
```yaml
# Override defaults (optional)
featurize:
  tile_size: 512
```

## Example Datasets

### CellPainting (Broad)
```bash
bioimage-embed featurize \
  --input "s3://cellpainting-gallery/cpg0016-jump/..." \
  --model dinov2-vit-base \
  --channels DAPI,Tubulin,Actin,Mito,ER \
  --output jump_features.parquet
```

### JUMP Dataset Reference
- Source: https://jump-cellpainting.broadinstitute.org/
- ~150M images, 116k perturbations
- Standard benchmark for biological embeddings

## Dependencies

```toml
[project]
dependencies = [
    "torch>=2.0",
    "transformers",      # HuggingFace
    "timm",              # Image models
    "hydra-core",
    "hydra-zen",
    "dask[distributed]",
    "pandas",
    "pyarrow",
    "anndata",
    "zarr",
    "tifffile",
    "typer",
    "rich",
]
```

## Milestones

### v0.1 - Featurization MVP
- [ ] CLI with `featurize` command
- [ ] DINOv2/CLIP/ResNet backbone support
- [ ] Dask tiling for large images
- [ ] Parquet output

### v0.2 - Training
- [ ] SimCLR pretraining
- [ ] Supervised fine-tuning
- [ ] Few-shot support

### v0.3 - Analysis
- [ ] UMAP/clustering
- [ ] CellPaint morphology profiles
- [ ] JUMP dataset examples

### v1.0 - Production
- [ ] Full documentation
- [ ] PyPI release
- [ ] Docker image

## Non-Goals

- ❌ Segmentation (use cellpose, stardist)
- ❌ Full training framework (use lightning)
- ❌ Complex YAML configs
- ❌ GUI

## References

- [JUMP Cell Painting](https://jump-cellpainting.broadinstitute.org/)
- [pycytominer](https://github.com/cytomining/pycytominer) - morphology feature extraction
- [DINOv2](https://github.com/facebookresearch/dinov2) - self-supervised vision
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) - biomedical vision-language
