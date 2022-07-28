import pytest
import os
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#  %%
from ast import excepthandler
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
import torch
from torch import nn
from pytorch_lightning import loggers as pl_loggers
import torchvision
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage import convolve, sobel
from skimage.measure import find_contours
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchinfo import summary

from mask_vae.datasets import DSB2018
from mask_vae.transforms import CropCentroidPipeline, MaskToDistogramPipeline, DistogramToMaskPipeline
from mask_vae.transforms import ImagetoDistogram, cropCentroid, DistogramToCoords, DistogramToCoords

from mask_vae.models import AutoEncoder, VQ_VAE, Mask_VAE, VAE
from mask_vae.lightning import LitAutoEncoderTorch, LitAutoEncoderPyro

interp_size = 128*4

window_size = 96
batch_size = 32
num_training_updates = 15000

num_hiddens = 64
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

train_dataset_glob = "data/stage1_train/*/masks/*.png"
train_dataset_glob = "data/BBBC010_v1_foreground_eachworm/*.png"

# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")

transformer_crop = CropCentroidPipeline(window_size)
transformer_dist = MaskToDistogramPipeline(window_size, interp_size)
transformer_coords = DistogramToCoords(window_size)

train_dataset_raw = DSB2018(train_dataset_glob)
train_dataset_crop = DSB2018(
    train_dataset_glob, transform=CropCentroidPipeline(window_size))
train_dataset_dist = DSB2018(train_dataset_glob, transform=transformer_dist)

# img_squeeze = train_dataset_crop[1].unsqueeze(0)
img_crop = train_dataset_crop[1].unsqueeze(0)

train_dataset = train_dataset_dist
test_img = train_dataset_dist[1].unsqueeze(0)


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8, pin_memory=True, collate_fn=my_collate)

# def test_transforms():
#     dist = np.array(train_dataset_crop[1][0]).astype(float)
#     plt.imshow(dist)
#     plt.show()


class TestVAE():

    def setup(self):
        # self.model2 = VAE(1, 10)
        self.model = VAE(3, 10)

    # def test_summary(self):
    #     print(summary(self.model, (1, 64, 64), device='cpu'))
    #     # print(summary(self.model2, (3, 64, 64), device='cpu'))

    def test_forward(self):
        x = torch.randn(16, 3, 64, 64)
        y = self.model(x)
        print("Model Output size:", y[0].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    def test_loss(self):
        x = torch.randn(16, 3, 64, 64)

        result = self.model(x)
        loss = self.model.loss_function(*result, M_N=0.005)
        print(loss)


def test_dist_to_coord():
    # dist = transformer_dist(train_dataset[0][0])
    plt.close()
    # TODO Faulty?
    coords = DistogramToCoords(window_size)(test_img)
    plt.scatter(coords[0][0][:, 0], coords[0][0][:, 1])
    plt.savefig("tests/test_dist_to_coord.png")
    plt.show()


def test_pipeline_forward():
    plt.close()
    # dist = MaskToDistogramPipeline(window_size)(train_dataset_raw[0])
    # plt.imshow(dist)
    # plt.savefig("tests/test_mask_to_dist.png")
    # plt.show()
    # plt.close()
    dist = test_img
    plt.imshow(dist.squeeze())
    plt.savefig("tests/test_pipeline_forward.png")
    plt.show()
    mask = DistogramToMaskPipeline(window_size)(dist)
    plt.imshow(mask.squeeze())
    plt.savefig("tests/test_dist_to_mask.png")
    plt.show()


def test_dist_to_coord():
    plt.close()
    # dist = transformer_dist(train_dataset[0][0])
    coords = transformer_coords(test_img)
    plt.scatter(coords[0][:, 0], coords[0][:, 1])
    plt.savefig("tests/test_dist_to_coord.png")
    plt.show()


@pytest.mark.parametrize("model", [VQ_VAE(channels=1),
                                   VAE(1, 64, image_dims=(
                                       interp_size, interp_size))])
class TestModels:
    def setup(self,model):
        pass
        
    def test_models(self,model):
        # vae = AutoEncoder(1, 1)
        # vae = VQ_VAE(channels=1)
        img = test_img
        # loss, x_recon, perplexity = model(img)
        result = model(img)
        z, log_var = model.encode(img)
        y_prime = model.decode(z)
        # print(f"img_dims:{img.shape} y:_dims:{x_recon.shape}")
        print(
            f"img_dims:{img.shape}, z:_dims:{z.shape}")


    def test_forward(self,model):
        model = Mask_VAE(model)
        # test_img = train_dataset[0]
        z = model.encode(test_img)
        y_prime = model.decode(z)
        model.forward(test_img)


    def test_training(self,model):
        model = Mask_VAE(model)
        lit_model = LitAutoEncoderTorch(model)
        trainer = pl.Trainer(
            max_steps=1,
            # limit_train_batches=1,
            # limit_val_batches=1
            # logger=tb_logger,
            # enable_checkpointing=True,
            # gpus=1,
            # accumulate_grad_batches=1,
            # callbacks=[checkpoint_callback],
            # min_epochs=1,
            max_epochs=1,
        )  # .from_argparse_args(args)
        trainer.fit(lit_model, dataloader)

# @pytest.mark.skipif(sys.version_info < (3,3))
# def test_model(model):
#     for i in range(10):
#         z_random = torch.normal(torch.zeros_like(z), torch.ones_like(z)).cuda()
#         generated_image = model.autoencoder.decoder(z_random)
#         plt.imshow(transforms.ToPILImage()(generated_image[0]))
#         plt.show()

# def test_mask_vae():
#     MaskVAE(VQ_VAE(channels=1))

