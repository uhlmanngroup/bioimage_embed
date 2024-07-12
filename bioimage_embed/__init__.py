from .bioimage_embed import BioImageEmbed
from .models import ModelFactory, create_model
from .config import Config
from . import augmentations
from .lightning import AutoEncoderSupervised, AutoEncoderUnsupervised
import torch

torch.cuda.empty_cache()
# from . import models, lightning, cli, export, config
# TODO: Fix this import as it currently produces too many warnings
# import logging
# logging.captureWarnings(True)

# Defining the public API
__all__ = [
    "AutoEncoderSupervised",
    "AutoEncoderUnsupervised",
    "BioImageEmbed",
    "Config",
    "augmentations",
    "ModelFactory",
    "create_model",
]
