# bioimage_embed - Project Specification

> Thin, maintainable wrapper for biological image embeddings. Library-first, CLI via Hydra-zen.

## Design Constraints

1. **Library-first** — importable modules, CLI emerges from schema
2. **uv** — package management, no poetry
3. **HuggingFace ecosystem** — Trainer, datasets, transformers (not Lightning)
4. **Hydra-zen** — Python schemas define CLI automatically, minimal YAML
5. **Lean on existing schemas** — HF TrainingArguments, timm configs, etc.
6. **Atomic PRs** — small, testable, sanity-checked units
7. **Worktrees for parallel work** — multiple workers simultaneously

## Architecture

```
bioimage_embed/
├── __init__.py           # Public API
├── config.py             # Hydra-zen schemas (Python, not YAML)
├── models/
│   └── backbones.py      # Thin wrapper: timm + transformers
├── data/
│   ├── datasets.py       # HF datasets integration
│   └── transforms.py     # albumentations/torchvision
├── train/
│   └── trainer.py        # HF Trainer wrapper
├── inference/
│   └── embed.py          # Batch embedding extraction
└── io/
    └── formats.py        # parquet, zarr, h5ad writers
```

## Config Philosophy

**CLI from schema, not manual argparse:**

```python
# config.py
from dataclasses import dataclass
from hydra_zen import builds, store, zen
from transformers import TrainingArguments

@dataclass
class EmbedConfig:
    model: str = "facebook/dinov2-base"
    batch_size: int = 32
    output_format: str = "parquet"

# Reuse HF's schema
TrainConfig = builds(TrainingArguments, populate_full_signature=True)

# CLI auto-generated
store(EmbedConfig, name="embed")
store(TrainConfig, name="train")
```

**Usage emerges automatically:**
```bash
python -m bioimage_embed.embed model=dinov2 batch_size=64
python -m bioimage_embed.train learning_rate=1e-4 num_train_epochs=10
```

## Dependencies

```toml
[project]
requires-python = ">=3.10"
dependencies = [
    "torch>=2.0",
    "transformers>=4.35",
    "datasets",
    "timm",
    "hydra-core",
    "hydra-zen",
    "albumentations",
    "pandas",
    "pyarrow",
    "zarr",
    "tifffile",
]

[project.optional-dependencies]
train = ["accelerate", "wandb"]
bio = ["anndata", "scanpy"]
```

## Refactor Plan (Atomic Issues)

### Phase 1: Foundation
- [ ] #4: Migrate poetry → uv, update pyproject.toml
- [ ] #5: Remove Lightning, add HF Trainer skeleton
- [ ] #6: Hydra-zen config schemas (reuse HF TrainingArguments)
- [ ] #7: timm/transformers backbone wrapper

### Phase 2: Core Features  
- [ ] #8: HF datasets integration for microscopy
- [ ] #9: Embedding extraction with Dask batching
- [ ] #10: Output writers (parquet, zarr, h5ad)

### Phase 3: Training
- [ ] #11: Self-supervised training (DINO/SimCLR via HF)
- [ ] #12: Fine-tuning with HF Trainer
- [ ] #13: Few-shot learning

### Phase 4: Examples
- [ ] #14: CellPaint example notebook
- [ ] #15: JUMP dataset integration

## HPC Considerations

- **Dask** for tiling large images
- **accelerate** for multi-GPU via HF
- **SLURM** compatible (hydra submitit launcher)
- **zarr** for chunked storage

## Non-Goals

- ❌ Custom training loops (use HF Trainer)
- ❌ GUI
- ❌ Segmentation (use cellpose)
- ❌ Complex YAML configs

## References

- [HuggingFace Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Hydra-zen](https://mit-ll-responsible-ai.github.io/hydra-zen/)
- [timm](https://huggingface.co/docs/timm/)
- [JUMP Cell Painting](https://jump-cellpainting.broadinstitute.org/)
