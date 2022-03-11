#  %%
from mask_vae.transforms import cropCentroid, DistogramtoImage, ImagetoDistogram
from mask_vae.lightning import LitAutoEncoder, LitVariationalAutoEncoder
from mask_vae.models import AutoEncoder, VQ_VAE, VAE
from mask_vae.datasets import DSB2018
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


#  %%

window_size = 128-32
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


transformer_crop = transforms.Compose(
    [
        # transforms.ToPILImage(),
        cropCentroid(window_size),
        transforms.ToTensor(),
        # transforms.Normalize(0, 1),
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
        # transforms.RandomCrop((512, 512)),
        # transforms.ConvertImageDtype(torch.bool)

    ]
)

transformer_dist = transforms.Compose(
    [
        transformer_crop,
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        ImagetoDistogram(window_size),
        # transforms.ToPILImage(),
        # transforms.RandomCrop((512, 512)),
        transforms.ConvertImageDtype(torch.float32)
    ]
)

transformer_image = transforms.Compose(
    [
        DistogramtoImage(window_size)
    ]
)

transformer = transformer_dist


train_dataset_glob = os.path.join(os.path.expanduser("~"),
                                  "data-science-bowl-2018/stage1_train/*/masks/*.png")
# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")
train_dataset_dist = DSB2018(train_dataset_glob, transform=transformer_dist)
train_dataset_crop = DSB2018(train_dataset_glob, transform=transformer_crop)
plt.imshow(train_dataset_crop[0][0])
plt.show()

train_dataset = train_dataset_dist
#  %%


transform_disttoimage = transforms.Compose([
    DistogramtoImage(window_size)
])


# dist = transformer_dist(train_dataset[0][0])
coords = transform_disttoimage(train_dataset_dist[0])
plt.scatter(coords[0][:, 0], coords[0][:, 1])
plt.show()
# print(out.shape)
# plt.imshow(transforms.ToPILImage()(transformer(train_dataset[0])))
# cell_image = transforms.ToPILImage()(train_dataset[0])
# distogram = ImagetoDistogram(512)(cell_image).astype(np.float32)
# plt.imshow(transforms.ToPILImage()(distogram))
# x = np.array(cell_image)
# out = convolve(x,fil, mode='constant')


# plt.imshow(cell_image)
# plt.show()
#  %%


# plt.plot(rho)
# plt.plot(rho_interp)
# plt.show()
# plt.plot(phi)
# plt.plot(phi_interp)
# plt.show()
# #  %%

# xii,yii = pol2cart(rho_interp,phi_interp)
# plt.plot(xii,yii)
# plt.show()
# #  %%
# xxii,yyii = np.meshgrid(xii,yii)

# dist_euclid = euclidean_distances(np.array([xii,yii]).T)
# plt.imshow(dist_euclid)
# plt.show()

# plt.plot(X_transform[:,0],X_transform[:,1])
#  %%
# batch_size = 32

dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8, pin_memory=True)

#  %%

# fig,ax = plt.subplots(10,10)
# for i,ax in enumerate(ax.flat):
#     ax.imshow(transform(train_dataset[i]).reshape(window_size,window_size))

#  %%


# vae = AutoEncoder(1, 1)
vae = VQ_VAE(channels=1)
img = train_dataset[0].unsqueeze(0)
loss, x_recon, perplexity = vae(img)
z = vae.encoder(img)
y_prime = vae.decoder(z)

# print(f"img_dims:{img.shape} y:_dims:{x_recon.shape}")
print(f"img_dims:{img.shape} x_recon:_dims:{x_recon.shape} z:_dims:{z.shape}")

# %%


# %%


tb_logger = pl_loggers.TensorBoardLogger("runs/")

# from pathlib import Path
# Path("checkpoints/").mkdir(parents=True, exist_ok=True)

# checkpoint_callback = ModelCheckpoint(
#     dirpath="checkpoints/",
# )

trainer = pl.Trainer(
    # default_root_dir="checkpoints/",
    logger=tb_logger,
    # enable_checkpointing=True,
    gpus=1,
    accumulate_grad_batches=1,
    # callbacks=[checkpoint_callback],
    min_epochs=50,
    max_epochs=75,
)  # .from_argparse_args(args)

model = LitAutoEncoder(VQ_VAE(channels=1))
# LitAutoEncoder.load_from_checkpoint(f'{checkpoint_callback.dirpath}/last.ckpt')
#  %%
# model = LitVariationalAutoEncoder()
trainer.fit(model,f'{checkpoint_callback.dirpath}/last.ckpt')

# tb_logger = pl_loggers.TensorBoardLogger("runs/")

# checkpoint_callback = ModelCheckpoint(
#             dirpath="checkpoints/",
#             save_last=True,
#             every_n_train_steps=100)

# last_checkpoint_path = "checkpoints/last.ckpt"

# if not(os.path.isfile(last_checkpoint_path)):
#     last_checkpoint_path = ""

# trainer = pl.Trainer(
#     resume_from_checkpoint=last_checkpoint_path,
#     logger=tb_logger,
#     enable_checkpointing=True,
#     gpus=1,
#     accumulate_grad_batches=1,
#     callbacks=[checkpoint_callback],
#     min_epochs=50,
#     max_epochs=75,
# )  # .from_argparse_args(args)

# #
# # if __name__ = main:
# #

# model = LitAutoEncoder(batch_size=batch_size)
# # model = LitVariationalAutoEncoder()
# trainer.fit(model, dataloader)

#  %%
# model
for i in range(10):
    z_random = torch.normal(torch.zeros_like(z), torch.ones_like(z)).cuda()
    generated_image = model.autoencoder.decoder(z_random)
    plt.imshow(transforms.ToPILImage()(generated_image[0]))
    plt.show()

# loss_function = torch.nn.MSELoss()
#  %%
# optimizer = torch.optim.Adam(model.parameters())
# epochs = 20
# outputs = []
# losses = []
# for epoch in range(epochs):
#     for image in train_dataloader:

#         # Reshaping the image to (-1, 784)
#     #   image = image.reshape(-1, 28*28)

#         # Output of Autoencoder
#         reconstructed = model(image)

#         # Calculating the loss function
#         loss = loss_function(reconstructed, image)

#         # The gradients are set to zero,
#         # the the gradient is computed and stored.
#         # .step() performs parameter update
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Storing the losses in a list for plotting
#         losses.append(loss)
#     outputs.append((epochs, image, reconstructed))

# # Defining the Plot Style
# plt.style.use('fivethirtyeight')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')

# # Plotting the last 100 values
# plt.plot(losses[-100:])
# from torchviz import make_dot
# make_dot(y,params=dict(model.named_parameters()))
#  %%


# class MaskAE(pl.LightningModule):
#         def __init__(self):
#         super(MaskAE, self).__init__()
#         self.batch_size = 4
#         self.learning_rate = 1e-3
# #         self.net = torchvision.models.segmentation.fcn_resnet50(pretrained = False, progress = True, num_classes = 19)
# #         self.net = UNet(num_classes = 19, bilinear = False)
# #         self.net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False, progress = True, num_classes = 19)
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             cropCentroid()
#         ])
#         self.trainset = None
#         self.testset = None

#     def forward(self, x):
#         return self.net(x)

#     def training_step(self, batch, batch_nb) :
#         img, mask = batch
#         img = img.float()
#         mask = mask.long()
#         out = self.forward(img)
#         loss_val = F.cross_entropy(out, mask, ignore_index = 250)
# #         print(loss.shape)
#         return {'loss' : loss_val}

#     def configure_optimizers(self):
#         opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
#         sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
#         return [opt], [sch]

#     def train_dataloader(self):
#         return DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True)

#     def test_dataloader(self):
#         return DataLoader(self.testset, batch_size = 1, shuffle = True)


#  %%


# #  %%
# class SegModel(pl.LightningModule):
#     def __init__(self):
#         super(SegModel, self).__init__()
#         self.batch_size = 4
#         self.learning_rate = 1e-3
# #         self.net = torchvision.models.segmentation.fcn_resnet50(pretrained = False, progress = True, num_classes = 19)
# #         self.net = UNet(num_classes = 19, bilinear = False)
# #         self.net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained = False, progress = True, num_classes = 19)
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             cropCentroid()
#         ])
#         self.trainset = None
#         self.testset = None

#     def forward(self, x):
#         return self.net(x)

#     def training_step(self, batch, batch_nb) :
#         img, mask = batch
#         img = img.float()
#         mask = mask.long()
#         out = self.forward(img)
#         loss_val = F.cross_entropy(out, mask, ignore_index = 250)
# #         print(loss.shape)
#         return {'loss' : loss_val}

#     def configure_optimizers(self):
#         opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
#         sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
#         return [opt], [sch]

#     def train_dataloader(self):
#         return DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True)

#     def test_dataloader(self):
#         return DataLoader(self.testset, batch_size = 1, shuffle = True)


#  %%
# test_dataloader_dir="data/stage1_test"

# val_dataloader_dir=
# test_dataloader_dir=
# predict_dataloader_dir=

# %%

# %%

# %%
