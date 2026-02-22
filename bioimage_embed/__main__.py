"""
CLI entry point for bioimage_embed using hydra-zen.

This module provides a CLI interface that is auto-generated from
the configuration schemas defined in config_zen.py.

Usage:
    python -m bioimage_embed --help
    python -m bioimage_embed train
    python -m bioimage_embed inference
    python -m bioimage_embed --config-name=base_config
"""

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from hydra_zen import make_config, zen, builds, to_yaml
import sys

from .config_zen import (
    BioImageEmbedConfig,
    EmbedConfig,
    DataConfig,
    AugmentationConfig,
    TrainerConfig,
    OptimizerConfig,
    SchedulerConfig,
    PathsConfig,
)


# ============================================================================
# CLI Commands
# ============================================================================

def train_command(cfg: DictConfig) -> None:
    """Train a bioimage embedding model."""
    print("=" * 80)
    print("TRAINING BIOIMAGE EMBEDDING MODEL")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("\nTraining not yet implemented - this is a demonstration of config generation")
    print("=" * 80)
    
    # TODO: Implement actual training
    # from . import BioImageEmbed
    # bie = BioImageEmbed(cfg)
    # bie.train()


def inference_command(cfg: DictConfig) -> None:
    """Run inference with a trained model."""
    print("=" * 80)
    print("RUNNING INFERENCE")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("\nInference not yet implemented - this is a demonstration of config generation")
    print("=" * 80)
    
    # TODO: Implement actual inference
    # from . import BioImageEmbed
    # bie = BioImageEmbed(cfg)
    # bie.infer()


def validate_command(cfg: DictConfig) -> None:
    """Validate configuration and dataset."""
    print("=" * 80)
    print("VALIDATING CONFIGURATION")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("\n✓ Configuration is valid")
    print("=" * 80)


# ============================================================================
# Hydra Configuration Store Setup
# ============================================================================

cs = ConfigStore.instance()

# Register base configuration
cs.store(name="base_config", node=BioImageEmbedConfig)

# Register sub-configurations for composition
cs.store(group="embed", name="base", node=EmbedConfig)
cs.store(group="data", name="base", node=DataConfig)
cs.store(group="augmentation", name="base", node=AugmentationConfig)
cs.store(group="trainer", name="base", node=TrainerConfig)
cs.store(group="optimizer", name="base", node=OptimizerConfig)
cs.store(group="scheduler", name="base", node=SchedulerConfig)
cs.store(group="paths", name="base", node=PathsConfig)


# ============================================================================
# Main CLI Entry Point
# ============================================================================

@hydra.main(version_base="1.3", config_path=None, config_name="base_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for bioimage_embed CLI.
    
    This function is decorated with @hydra.main which:
    - Auto-generates CLI arguments from the config schema
    - Handles config file loading and composition
    - Provides --help documentation
    - Enables config overrides via command line
    
    Examples:
        python -m bioimage_embed
        python -m bioimage_embed embed.model=resnet50
        python -m bioimage_embed data.batch_size=64 trainer.max_epochs=200
        python -m bioimage_embed --cfg job
        python -m bioimage_embed --help
    """
    
    # Determine command from config or args
    # For now, default to showing configuration
    print("=" * 80)
    print("BIOIMAGE EMBED - Hydra-zen Configuration")
    print("=" * 80)
    print("\nCurrent Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("\nAvailable commands:")
    print("  train       - Train a new model")
    print("  inference   - Run inference with trained model")
    print("  validate    - Validate configuration")
    print("\nTo run a command, add 'command=<name>' to your arguments")
    print("Example: python -m bioimage_embed command=train")
    print("=" * 80)
    
    # Check if command is specified
    if "command" in cfg:
        command = cfg.command
        if command == "train":
            train_command(cfg)
        elif command == "inference":
            inference_command(cfg)
        elif command == "validate":
            validate_command(cfg)
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)


# ============================================================================
# Standalone Command Entry Points
# ============================================================================

@hydra.main(version_base="1.3", config_path=None, config_name="base_config")
def train(cfg: DictConfig) -> None:
    """Standalone train command entry point."""
    train_command(cfg)


@hydra.main(version_base="1.3", config_path=None, config_name="base_config")
def inference(cfg: DictConfig) -> None:
    """Standalone inference command entry point."""
    inference_command(cfg)


@hydra.main(version_base="1.3", config_path=None, config_name="base_config")
def validate(cfg: DictConfig) -> None:
    """Standalone validate command entry point."""
    validate_command(cfg)


# ============================================================================
# Script Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
