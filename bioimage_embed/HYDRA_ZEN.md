# Hydra-zen Configuration Schemas

This PR adds hydra-zen based configuration schemas to bioimage_embed, providing a Python-first configuration approach with auto-generated CLI interfaces.

## Changes Made

### 1. Dependencies Added

Added `hydra-zen = "^0.13.0"` to `pyproject.toml` (hydra-core ^1.3.2 was already present).

### 2. New Configuration Module (`bioimage_embed/config_zen.py`)

Created a new configuration module using hydra-zen's `builds()` and `store()` patterns:

#### Configuration Schemas

- **EmbedConfig**: Core embedding configuration (model, batch_size, latent_dim, etc.)
- **DataConfig**: Dataset and dataloader settings
- **AugmentationConfig**: Image augmentation parameters
- **TrainerConfig**: PyTorch Lightning trainer settings
- **OptimizerConfig**: Optimizer configuration
- **SchedulerConfig**: Learning rate scheduler configuration
- **PathsConfig**: Output paths configuration
- **BioImageEmbedConfig**: Main application configuration composing all sub-configs

#### HuggingFace Integration

Used `builds()` to wrap HuggingFace's `TrainingArguments`:

```python
TrainingArgumentsConfig = builds(
    TrainingArguments,
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    # ... other parameters
    populate_full_signature=True,
)
```

This allows reusing HF's training API with Hydra configuration.

### 3. CLI Entry Point (`bioimage_embed/__main__.py`)

Created a new CLI entry point that:

- Auto-generates command-line arguments from config schemas
- Provides `--help` documentation
- Enables config overrides via command line
- Supports config file loading and composition

### 4. Benefits

**Python-First Configuration**
- Type-safe dataclasses instead of YAML files
- IDE auto-completion and type checking
- Validation at definition time

**Auto-Generated CLI**
- No manual argparse code
- CLI automatically stays in sync with config schemas
- Built-in help documentation

**Composable Configs**
- Modular configuration groups
- Easy to override specific settings
- Supports config inheritance

**Integration with Existing Tools**
- Wraps HuggingFace TrainingArguments
- Compatible with PyTorch Lightning
- Works with existing albumentations pipelines

## Installation

```bash
# Install dependencies
poetry install

# Or with pip
pip install hydra-core hydra-zen
```

## Usage

### Basic Usage

```bash
# Show current configuration
python -m bioimage_embed

# Show help (auto-generated from schemas)
python -m bioimage_embed --help

# Override configuration values
python -m bioimage_embed embed.model=resnet50

# Override multiple values
python -m bioimage_embed \
    embed.model=resnet50 \
    data.batch_size=64 \
    trainer.max_epochs=200

# Show job configuration
python -m bioimage_embed --cfg job
```

### Running Commands

```bash
# Train a model
python -m bioimage_embed command=train

# Run inference
python -m bioimage_embed command=inference

# Validate configuration
python -m bioimage_embed command=validate
```

### Configuration Groups

Override entire configuration groups:

```bash
# Use different optimizer
python -m bioimage_embed optimizer=sgd

# Use different data config
python -m bioimage_embed data=imagenet
```

### Configuration Files

Generate a config file:

```bash
# Print configuration as YAML
python -m bioimage_embed --cfg job > config.yaml

# Use custom config file
python -m bioimage_embed --config-path=/path/to/configs --config-name=my_config
```

## Testing

A test script is provided to verify the configuration schemas:

```bash
python test_config_zen.py
```

Expected output:
```
Testing configuration schemas...
================================================================================
✓ EmbedConfig: model=facebook/dinov2-base, batch_size=32
✓ DataConfig: data_path=data, num_workers=4
✓ AugmentationConfig: enable=True
✓ TrainerConfig: max_epochs=125, accelerator=auto
✓ OptimizerConfig: optimizer=adamw, lr=0.0001
✓ SchedulerConfig: scheduler=cosine, min_lr=1e-06
✓ PathsConfig: model_dir=models
✓ BioImageEmbedConfig: seed=42
================================================================================

✓ All configuration schemas successfully instantiated!

Default EmbedConfig values:
  model: facebook/dinov2-base
  batch_size: 32
  latent_dim: 64
  pretrained: True
```

## Migration from Old Config

The existing `config.py` and `cli.py` remain untouched. This is an additive change:

- Old config system still works
- New hydra-zen system provides modern alternative
- Can gradually migrate or run both in parallel

To integrate with existing `BioImageEmbed` class:

```python
from bioimage_embed import BioImageEmbed
from bioimage_embed.config_zen import BioImageEmbedConfig
from omegaconf import OmegaConf

# Create config
cfg = BioImageEmbedConfig()

# Convert to OmegaConf for compatibility
ocfg = OmegaConf.structured(cfg)

# Use with existing class
bie = BioImageEmbed(ocfg)
bie.train()
```

## Future Enhancements

1. **Full Integration**: Update `BioImageEmbed` class to natively use hydra-zen configs
2. **More Config Groups**: Add presets for common model architectures and datasets
3. **Validation**: Add pydantic validators for complex constraints
4. **Multirun**: Leverage Hydra's multirun for hyperparameter sweeps
5. **Structured Logging**: Better integration with experiment tracking

## References

- [Hydra documentation](https://hydra.cc/)
- [Hydra-zen documentation](https://mit-ll-responsible-ai.github.io/hydra-zen/)
- [HuggingFace TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)
