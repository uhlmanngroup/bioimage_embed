# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pythae
import pytorch_lightning as pl
import torch
from PIL import Image
from pythae import models
from pythae.data.datasets import DatasetOutput
from pythae.models import VAE, VAEConfig
from pythae.models.nn.benchmarks import celeba
from pythae.models.nn.benchmarks.cifar import (Decoder_ResNet_AE_CIFAR,
                                               Encoder_ResNet_VAE_CIFAR)
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
# Note - you must have torchvision installed for this example
from pytorch_lightning import loggers as pl_loggers
#  %%
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset

from bio_vae.datasets import DatasetGlob
from bio_vae.lightning import DatamoduleGlob, LitAutoEncoderTorch
from bio_vae.models import Bio_VAE, vae, vq_vae

Image.MAX_IMAGE_PIXELS = None

max_epochs = 500

window_size = 64 * 2
batch_size = 32
num_workers = 2**4
# model_name = "VQVAE"
dataset = "ivy_gap"
data_dir = "data"
train_dataset_glob = f"{data_dir}/{dataset}/random/*png"
learning_rate = 1e-3
optimizer_cls = "AdamW"
optimizer_params = {"weight_decay": 0.05, "betas": (0.91, 0.995)}
scheduler_cls = "ReduceLROnPlateau"
scheduler_params = {"patience": 5, "factor": 0.5}

num_embeddings = 512
decay = 0.99
learning_rate = 1e-3

channels = 3

input_dim = (channels, window_size, window_size)
latent_dim = 64


train_dataset = DatasetGlob(train_dataset_glob)

transform = transforms.Compose(
    [
        # transforms.Grayscale(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine((0, 360)),
        transforms.RandomResizedCrop(size=window_size),
        # transforms.RandomCrop(size=(512,512)),
        # transforms.GaussianBlur(5),
        transforms.ToTensor(),
        # transforms.Normalize((0.485), (0.229)),
    ]
)


train_dataset = DatasetGlob(train_dataset_glob, transform=transform)
# train_dataset = DatasetGlob(train_dataset_glob, transform=transform)

# plt.imshow(train_dataset[100][0], cmap="gray")
# plt.show()
# plt.close()

# %%
dataloader = DatamoduleGlob(
    train_dataset_glob,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    transform=transform,
    pin_memory=True,
    persistent_workers=True,
)


# model = Bio_VAE("VQ_VAE", channels=3, num_residual_layers=8, num_residual_hiddens=64)


model_config = models.VAEConfig(input_dim=(channels, window_size, window_size), latent_dim=latent_dim)


model = models.VAE(
    model_config=model_config,
    encoder=vae.Encoder(model_config),
    decoder=vae.Decoder(model_config),
)

model_config = models.VAEConfig(
    input_dim=(3, window_size, window_size), latent_dim=window_size
)

# model = models.VAE(
#     model_config=model_config,
#     encoder=celeba.Encoder_ResNet_VAE_CELEBA(model_config),
#     decoder=celeba.Decoder_ResNet_AE_CELEBA(model_config),
# )

# model = models.VAE(
#     model_config=model_config,
#     encoder=celeba.Encoder_Conv_VAE_CELEBA(model_config),
#     decoder=celeba.Decoder_Conv_AE_CELEBA(model_config),
# )

# model_config = VQVAEConfig(
#     input_dim=(3, window_size, window_size), latent_dim=window_size,
# )

# model = VQVAE(
#     model_config=model_config,
#     encoder=celeba.Encoder_ResNet_VQVAE_CELEBA(model_config),
#     decoder=celeba.Decoder_ResNet_VQVAE_CELEBA(model_config),
# )

# model = vae.VAE()

model_config_vqvae = pythae.models.VQVAEConfig(
    input_dim=input_dim, latent_dim=latent_dim, num_embeddings=num_embeddings
)


model = Bio_VAE(
    "VAE", model_config=model_config,in_channels=3, latent_dim=window_size, image_dims=(window_size, window_size)
)

model = pythae.models.VQVAE(
    model_config_vqvae,
    encoder=vq_vae.Encoder(
        model_config_vqvae,
    ),
    decoder=vq_vae.Decoder(
        model_config_vqvae,
    ),
)

model = Bio_VAE("VQ_VAE",model_config=model_config_vqvae, channels=channels)

# %%
# model = Bio_VAE("VQ_VAE", channels=3)


training_config = BaseTrainerConfig(
    output_dir="my_model",
    num_epochs=50,
    learning_rate=learning_rate,
    per_device_train_batch_size=200,
    per_device_eval_batch_size=200,
    steps_saving=None,
    optimizer_cls=optimizer_cls,
    optimizer_params=optimizer_params,
    scheduler_cls=scheduler_cls,
    scheduler_params=scheduler_params,
)


pipeline = TrainingPipeline(training_config=training_config, model=model)

dataloader.setup()
model
model.eval()
# model.forward({"data": dataloader.train_dataloader().dataset[0].unsqueeze(0)})
model.forward({"data":dataloader.train_dataloader().dataset[0].unsqueeze(0)})

# from tqdm import tqdm

# train_data = [data for data in tqdm(dataloader.train_dataloader())]
# eval_data = [data for data in tqdm(dataloader.test_dataloader())]

# train_data[0]


# # for i in tqdm(range(100)):
# #     for data in tqdm(train_data):
# #         train_data.append(transform(data))
# #     for data in tqdm(eval_data):
# #         eval_data.append(transform(data))

# train_data_big = torch.cat(train_data)
# eval_data_big = torch.cat(eval_data)

# %%


class AECustom(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.size = len(self.dataset)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        X = self.dataset[index]
        return DatasetOutput(data=X)


# pipeline(
#     train_data=AECustom(dataloader.train_dataloader().dataset),
#     eval_data=AECustom(dataloader.test_dataloader().dataset),
# )

model_name = model._get_name()
model_dir = f"models/{dataset}_{model_name}"

# %%
lit_model = LitAutoEncoderTorch(
    model,
    learning_rate=learning_rate,
    optimizer_cls=optimizer_cls,
    optimizer_params=optimizer_params,
    scheduler_cls=scheduler_cls,
    scheduler_params=scheduler_params,
)

tb_logger = pl_loggers.TensorBoardLogger(f"{model_dir}/runs/")

Path(f"{model_dir}/").mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(dirpath=f"{model_dir}/", save_last=True)

trainer = pl.Trainer(
    logger=tb_logger,
    gradient_clip_val=0.5,
    enable_checkpointing=True,
    gpus=1,
    accumulate_grad_batches=1,
    callbacks=[checkpoint_callback],
    min_epochs=50,
    max_epochs=max_epochs,
)  # .from_argparse_args(args)

# %%

try:
    trainer.fit(lit_model, datamodule=dataloader, ckpt_path=f"{model_dir}/last.ckpt")
except:
    trainer.fit(lit_model, datamodule=dataloader)
