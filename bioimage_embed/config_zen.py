"""
Hydra-zen based configuration schemas for bioimage_embed.

This module uses hydra-zen to create Python-first configuration schemas
that auto-generate CLI interfaces and Hydra configs.
"""

from dataclasses import dataclass
from typing import Optional, List
from hydra_zen import builds, store, make_config, zen
from transformers import TrainingArguments
import torch

# ============================================================================
# Core Configuration Schemas
# ============================================================================

@dataclass
class EmbedConfig:
    """Core embedding configuration."""
    model: str = "facebook/dinov2-base"
    batch_size: int = 32
    latent_dim: int = 64
    input_dim: List[int] = None
    pretrained: bool = True
    
    def __post_init__(self):
        if self.input_dim is None:
            self.input_dim = [3, 224, 224]


@dataclass
class DataConfig:
    """Dataset and dataloader configuration."""
    data_path: str = "data"
    num_workers: int = 4
    batch_size: int = 32
    dataset_type: str = "ImageFolder"  # ImageFolder, Tiff, Ngff, Fake
    

@dataclass
class AugmentationConfig:
    """Image augmentation configuration."""
    enable_augmentation: bool = True
    resize_height: int = 224
    resize_width: int = 224
    normalize: bool = True
    horizontal_flip_prob: float = 0.5
    

@dataclass
class TrainerConfig:
    """PyTorch Lightning trainer configuration."""
    max_epochs: int = 125
    min_epochs: int = 1
    accelerator: str = "auto"
    devices: str = "auto"
    num_nodes: int = 1
    gradient_clip_val: float = 0.5
    accumulate_grad_batches: int = 16
    log_every_n_steps: int = 1
    enable_checkpointing: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 0.001
    momentum: float = 0.9


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    t_initial: int = 10
    t_mul: int = 2
    cooldown_epochs: int = 5


@dataclass
class PathsConfig:
    """Output paths configuration."""
    model_dir: str = "models"
    logs_dir: str = "logs"
    tensorboard_dir: str = "tensorboard"
    wandb_dir: str = "wandb"


# ============================================================================
# Hydra-zen builds for external libraries
# ============================================================================

# Wrap HuggingFace TrainingArguments with hydra-zen
# This allows us to use HF's training API with Hydra configuration
TrainingArgumentsConfig = builds(
    TrainingArguments,
    output_dir="./outputs",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=500,
    load_best_model_at_end=True,
    populate_full_signature=True,
)


# ============================================================================
# Main Application Config
# ============================================================================

@dataclass
class BioImageEmbedConfig:
    """Main configuration for bioimage_embed application."""
    embed: EmbedConfig = None
    data: DataConfig = None
    augmentation: AugmentationConfig = None
    trainer: TrainerConfig = None
    optimizer: OptimizerConfig = None
    scheduler: SchedulerConfig = None
    paths: PathsConfig = None
    seed: int = 42
    
    def __post_init__(self):
        # Initialize sub-configs with defaults if not provided
        if self.embed is None:
            self.embed = EmbedConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.augmentation is None:
            self.augmentation = AugmentationConfig()
        if self.trainer is None:
            self.trainer = TrainerConfig()
        if self.optimizer is None:
            self.optimizer = OptimizerConfig()
        if self.scheduler is None:
            self.scheduler = SchedulerConfig()
        if self.paths is None:
            self.paths = PathsConfig()


# ============================================================================
# Hydra ConfigStore Registration
# ============================================================================

# Store configurations in Hydra's config store
# This makes them available for CLI and config file generation
store(EmbedConfig, name="embed", group="embed")
store(DataConfig, name="data", group="data")
store(AugmentationConfig, name="augmentation", group="augmentation")
store(TrainerConfig, name="trainer", group="trainer")
store(OptimizerConfig, name="optimizer", group="optimizer")
store(SchedulerConfig, name="scheduler", group="scheduler")
store(PathsConfig, name="paths", group="paths")
store(TrainingArgumentsConfig, name="hf_training", group="training")
store(BioImageEmbedConfig, name="base_config")


# ============================================================================
# Config Factory Functions
# ============================================================================

def make_train_config(**overrides):
    """Create a training configuration with optional overrides."""
    cfg = make_config(
        embed=EmbedConfig,
        data=DataConfig,
        augmentation=AugmentationConfig,
        trainer=TrainerConfig,
        optimizer=OptimizerConfig,
        scheduler=SchedulerConfig,
        paths=PathsConfig,
    )
    return cfg


def make_inference_config(**overrides):
    """Create an inference configuration with optional overrides."""
    cfg = make_config(
        embed=EmbedConfig,
        data=DataConfig,
        augmentation=AugmentationConfig,
        paths=PathsConfig,
    )
    return cfg
