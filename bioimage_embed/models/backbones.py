"""
Thin wrapper for loading pretrained backbone models from timm and HuggingFace transformers.

This module provides a unified interface for loading various pretrained models
including DINOv2, CLIP, ResNet, and ViT variants from both the timm and
transformers model hubs.
"""

import timm
import torch.nn as nn
from transformers import AutoModel


def load_backbone(name: str) -> nn.Module:
    """
    Load a pretrained backbone model from timm or HuggingFace transformers.

    This function provides a unified interface for loading models from both
    the timm library and HuggingFace transformers. Models are loaded with
    pretrained weights.

    Args:
        name: Model identifier. Can be:
            - A timm model name (e.g., "resnet50", "vit_base_patch16_224")
            - A HuggingFace model identifier starting with "facebook/" or "openai/"
              (e.g., "facebook/dinov2-base", "openai/clip-vit-base-patch32")

    Returns:
        A PyTorch nn.Module containing the loaded pretrained model.
        For timm models, num_classes is set to 0 to return features only.

    Examples:
        >>> # Load a ResNet50 from timm
        >>> model = load_backbone("resnet50")
        
        >>> # Load DINOv2 from HuggingFace
        >>> model = load_backbone("facebook/dinov2-base")
        
        >>> # Load CLIP from HuggingFace
        >>> model = load_backbone("openai/clip-vit-base-patch32")
    """
    # Check if this is a HuggingFace model identifier
    if name.startswith("facebook/") or name.startswith("openai/"):
        # Load from HuggingFace transformers
        return AutoModel.from_pretrained(name)
    
    # Otherwise, load from timm
    # Set num_classes=0 to get feature extraction mode (no classification head)
    return timm.create_model(name, pretrained=True, num_classes=0)
