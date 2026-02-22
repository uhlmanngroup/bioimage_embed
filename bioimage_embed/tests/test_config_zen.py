#!/usr/bin/env python
"""
Test script for hydra-zen configuration schemas.

This script tests the config_zen module without requiring
the full bioimage_embed dependencies.
"""

import sys
import os

# Import config_zen directly to avoid importing torch
import importlib.util
spec = importlib.util.spec_from_file_location(
    "config_zen",
    os.path.join(os.path.dirname(__file__), "bioimage_embed", "config_zen.py")
)
config_zen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_zen)

# Now we can use the classes
EmbedConfig = config_zen.EmbedConfig
DataConfig = config_zen.DataConfig
AugmentationConfig = config_zen.AugmentationConfig
TrainerConfig = config_zen.TrainerConfig
OptimizerConfig = config_zen.OptimizerConfig
SchedulerConfig = config_zen.SchedulerConfig
PathsConfig = config_zen.PathsConfig
BioImageEmbedConfig = config_zen.BioImageEmbedConfig

def test_configs():
    """Test that all config classes can be instantiated."""
    print("Testing configuration schemas...")
    print("=" * 80)
    
    # Test individual configs
    embed_cfg = EmbedConfig()
    print(f"✓ EmbedConfig: model={embed_cfg.model}, batch_size={embed_cfg.batch_size}")
    
    data_cfg = DataConfig()
    print(f"✓ DataConfig: data_path={data_cfg.data_path}, num_workers={data_cfg.num_workers}")
    
    aug_cfg = AugmentationConfig()
    print(f"✓ AugmentationConfig: enable={aug_cfg.enable_augmentation}")
    
    trainer_cfg = TrainerConfig()
    print(f"✓ TrainerConfig: max_epochs={trainer_cfg.max_epochs}, accelerator={trainer_cfg.accelerator}")
    
    opt_cfg = OptimizerConfig()
    print(f"✓ OptimizerConfig: optimizer={opt_cfg.optimizer}, lr={opt_cfg.lr}")
    
    sched_cfg = SchedulerConfig()
    print(f"✓ SchedulerConfig: scheduler={sched_cfg.scheduler}, min_lr={sched_cfg.min_lr}")
    
    paths_cfg = PathsConfig()
    print(f"✓ PathsConfig: model_dir={paths_cfg.model_dir}")
    
    # Test main config
    main_cfg = BioImageEmbedConfig()
    print(f"✓ BioImageEmbedConfig: seed={main_cfg.seed}")
    
    print("=" * 80)
    print("\n✓ All configuration schemas successfully instantiated!")
    print("\nDefault EmbedConfig values:")
    print(f"  model: {embed_cfg.model}")
    print(f"  batch_size: {embed_cfg.batch_size}")
    print(f"  latent_dim: {embed_cfg.latent_dim}")
    print(f"  pretrained: {embed_cfg.pretrained}")
    
    return True


if __name__ == "__main__":
    try:
        test_configs()
        sys.exit(0)
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
