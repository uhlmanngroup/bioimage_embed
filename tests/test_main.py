import pytest
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pythae

# Note - you must have torchvision installed for this example
from torch.utils.data import DataLoader
from bio_vae.models.legacy import vq_vae

from bio_vae.utils import collate_none, get_test_image

# from bio_vae.datasets import BroadDataset, DatasetGlob
from bio_vae.shapes.transforms import (
    CropCentroidPipeline,
    MaskToDistogramPipeline,
    DistogramToMaskPipeline,
)
from bio_vae.shapes.transforms import (
    DistogramToCoords,
)

# from bio_vae.models import VQ_VAE, Bio_VAE, VAE
from bio_vae.lightning import LitAutoEncoderTorch
from bio_vae.models.legacy import vae


# @pytest.mark.skip(reason="Crashes github actions")
models = [vq_vae.VQ_VAE, vae.VAE]


@pytest.mark.parametrize("model", models)
def test_training(self, model, dataset, wrapper):
    model = wrapper(model)
    lit_model = LitAutoEncoderTorch(model)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_none,
    )
    lit_model = LitAutoEncoderTorch(model)
    trainer = pl.Trainer(
        max_steps=1,
        max_epochs=1,
    )  # .from_argparse_args(args)
    # trainer.test(lit_model, dataloader)
    trainer.fit(lit_model, dataloader)
