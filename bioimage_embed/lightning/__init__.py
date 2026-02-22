"""
DEPRECATED: This module uses PyTorch Lightning which is being phased out.

Please use the new HuggingFace Trainer-based training in bioimage_embed.train instead.
See bioimage_embed.train.example_train for usage examples.

This module will be removed in a future version.
"""

import warnings

warnings.warn(
    "bioimage_embed.lightning is deprecated and will be removed in a future version. "
    "Please use bioimage_embed.train (HuggingFace Trainer) instead.",
    DeprecationWarning,
    stacklevel=2
)

from .pyro import LitAutoEncoderPyro
from .torch import AESupervised, AEUnsupervised, AutoEncoder, AE, AutoEncoderSupervised, AutoEncoderUnsupervised
from .dataloader import DataModule

__all__ = ["LitAutoEncoderPyro", "AESupervised", "AEUnsupervised", "DataModule", "AutoEncoder","AE","AutoEncoderUnsupervised","AutoEncoderSupervised"]
