import torch
import torch.nn.functional as F

# Note - you must have torchvision installed for this example
from torch.utils.data import DataLoader

from bio_vae.transforms import DistogramToMaskPipeline

# from .ae import AutoEncoder

# from .vae_bio import Mask_VAE, Image_VAE
# from .utils import BaseVAE
# from .legacy.vae import VAE
# from .vq_vae import VQ_VAE
from .legacy import Bio_VAE


from .bolts import ResNet18VAEEncoder, ResNet18VAEDecoder

import pythae

from . import bolts
from .factory import ModelFactory, create_model, MODELS



