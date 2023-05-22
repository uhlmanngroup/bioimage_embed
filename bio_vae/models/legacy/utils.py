import sys
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import pyro
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import glob

# Note - you must have torchvision installed for this example
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from skimage.measure import regionprops
from torchvision.transforms.functional import crop
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Callable, Union, Any, TypeVar, Tuple
from abc import abstractmethod

# from .types_ import *
from torch import nn
from abc import abstractmethod
from torch import Tensor

class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    # def encode(self, input: Tensor) -> List[Tensor]:
    #     raise NotImplementedError

    # def decode(self, input: Tensor) -> Any:
    #     raise NotImplementedError

    # def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
    #     raise NotImplementedError

    # def generate(self, x: Tensor, **kwargs) -> Tensor:
    #     raise NotImplementedError
    
    # def forward(self, x):
    #     return self.model(x)

    # def decoder(self, z):
    #     return self.model.decoder(z)

    # def encoder(self, img):
    #     return self.model.encoder(img)

    # def decode(self, z):
    #     return self.model.decode(z)

    # def encode(self, img):
    #     return self.model.encode(img)

    # def recon(self, img):
    #     return self.model.recon(img)

    # @abstractmethod
    # def forward(self, *inputs: Tensor) -> Tensor:
    #     pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    def output_from_results(self, *args, **kwargs):
        return self.model.output_from_results(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)