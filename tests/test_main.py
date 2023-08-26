import pytest
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pythae

# Note - you must have torchvision installed for this example
from torch.utils.data import DataLoader
from bioimage_embed.models.legacy import vq_vae

from bioimage_embed.utils import collate_none, get_test_image

# from bioimage_embed.datasets import BroadDataset, DatasetGlob
from bioimage_embed.shapes.transforms import (
    CropCentroidPipeline,
    MaskToDistogramPipeline,
    DistogramToMaskPipeline,
)
from bioimage_embed.shapes.transforms import (
    DistogramToCoords,
)
from bioimage_embed.models import MODELS

# from bioimage_embed.models import VQ_VAE, Bio_VAE, VAE
from bioimage_embed.lightning import LitAutoEncoderTorch
from bioimage_embed.models.legacy import vae
import  numpy as np
from bioimage_embed.models import create_model, MODELS


# @pytest.mark.skip(reason="Crashes github actions")
input_dim = [(3,64,64),(1,64,64)]
latent_dim = np.power(2,np.arange(2,5))
batch_size = [1,2]
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("input_dim", input_dim)
@pytest.mark.parametrize("latent_dim", latent_dim)
@pytest.mark.parametrize("batch_size", batch_size)
class TestModels:
    def test_training(self,model, input_dim, latent_dim,batch_size):
        # model = wrapper(model)
        dataset = torch.rand(batch_size,*input_dim)
        model = create_model(model, input_dim=input_dim, latent_dim=latent_dim)
        lit_model = LitAutoEncoderTorch(model)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=False,
            collate_fn=collate_none,
        )
        trainer = pl.Trainer(
            max_steps=1,
            max_epochs=1,
        )  # .from_argparse_args(args)
        # trainer.test(lit_model, dataloader)
        trainer.fit(lit_model, dataloader)

    def test_latent_space(self,model,input_dim,latent_dim,batch_size):
        dataset = torch.rand(batch_size,*input_dim)
        model = create_model(model, input_dim=input_dim, latent_dim=latent_dim)
        z = model({"data":dataset})["z"]
        assert tuple(z.shape) == (batch_size,latent_dim)