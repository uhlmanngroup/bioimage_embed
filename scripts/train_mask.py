# %%
import os
from pathlib import Path

from torch.autograd import Variable
import torchvision

#  %%
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch

# Note - you must have torchvision installed for this example
from torch.utils.data import DataLoader
import os
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from bio_vae.lightning import DataModuleGlob, DataModule

from torchvision import datasets
from bio_vae.datasets import DatasetGlob
from bio_vae.shapes.transforms import (
    CropCentroidPipeline,
    DistogramToCoords,
    DistogramToCoords,
    MaskToDistogramPipeline,
)

# from bio_vae.models import Mask_VAE, VQ_VAE, VAE
from bio_vae.lightning import LitAutoEncoderTorch, LitAutoEncoderPyro
import matplotlib.pyplot as plt

from bio_vae.lightning import DataModule, DataModuleGlob, LitAutoEncoderTorch
import matplotlib
matplotlib.use('TkAgg')
interp_size = 128 *2

max_epochs = 500

window_size = 128*2


params = {
    "epochs": 500,
    "batch_size": 4,
    "num_workers": 2**4,
    # "window_size": 64*2,
    "num_workers": 1,
    "input_dim": (1, window_size, window_size),
    # "channels": 3,
    "latent_dim": 16,
    "num_embeddings": 16,
    "num_hiddens": 16,
    "num_residual_hiddens": 32,
    "num_residual_layers": 150,
    # "embedding_dim": 32,
    # "num_embeddings": 16,
    "commitment_cost": 0.25,
    "decay": 0.99,
}

optimizer_params = {
    "opt": "LAMB",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "momentum": 0.9,
}

lr_scheduler_params = {
    "sched": "cosine",
    "min_lr": 1e-4,
    "warmup_epochs": 5,
    "warmup_lr": 1e-6,
    "cooldown_epochs": 10,
    "t_max": 50,
    "cycle_momentum": False,
}

# channels = 3
import argparse 
from types import SimpleNamespace
# input_dim = (params["channels"], params["window_size"], params["window_size"])
args = SimpleNamespace(**params, **optimizer_params, **lr_scheduler_params)

dataset = "bbbc010/BBBC010_v1_foreground_eachworm"
dataset = "bbbc010"
model_name = "vqvae"

train_data_path = f"data/{dataset}"


# train_dataset_glob = "data-science-bowl-2018/stage1_train/*/masks/*.png"
# train_dataset_glob = "data/stage1_train/*/masks/*.png"
# train_dataset_glob = f"data/{dataset}/*.png"
# %%
# train_dataset_glob = os.path.join("data/BBBC010_v1_foreground_eachworm/*.png")


# train_dataset_glob = os.path.join("data/DatasetGlob/train/masks/*.tif")
# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")

# model_dir = "test"
# model_dir = "BBBC010_v1_foreground_eachworm"
model_dir = f"models/{dataset}_{model_name}"
# %%

transform_crop = CropCentroidPipeline(window_size)
transform_dist = MaskToDistogramPipeline(window_size, interp_size)
transform_coords = DistogramToCoords(window_size)

transform_mask_to_gray = transforms.Compose([transforms.Grayscale(1)])

transform_mask_to_crop = transforms.Compose(
    [
        # transforms.ToTensor(),
        transform_mask_to_gray,
        transform_crop,
    ]
)

transform_mask_to_dist = transforms.Compose(
    [
        transform_mask_to_crop,
        transform_dist,
    ]
)
transform_mask_to_coords = transforms.Compose(
    [
        transform_mask_to_dist,
        # transform_coords,
    ]
)

# train_data = torchvision.datasets.ImageFolder(
# "/home/ctr26/gdrive/+projects/idr_autoencode_torch/data/bbbc010"
# )
# train_dataset_crop = DatasetGlob(
#     train_dataset_glob, transform=CropCentroidPipeline(window_size))
transforms_dict = {
    "none": transform_mask_to_gray,
    "transform_crop": transform_mask_to_crop,
    "transform_dist": transform_mask_to_dist,
    # "transform_coords": transform_mask_to_coords,
}

train_data = {
    key: datasets.ImageFolder(train_data_path, transform=value)
    for key, value in transforms_dict.items()
}

for key, value in train_data.items():
    print(key, len(value))
    plt.imshow(train_data[key][0][0], cmap="gray")
    plt.imsave(f"{key}.png", train_data[key][0][0], cmap="gray")
    # plt.show()
    plt.close()


# train_dataset = DatasetGlob(train_data, transform=transforms.ToTensor())
# plt.imshow(train_dataset[0][0], cmap="gray")
# plt.show()
# train_dataset = DatasetGlob(train_dataset_glob, transform=transformer_crop)
# plt.imshow(train_dataset[0][0], cmap="gray")
# plt.show()

# train_dataset = DatasetGlob(train_dataset_glob, transform=transformer_crop)
# plt.imshow(train_dataset[0][0], cmap="gray")
# plt.show()

# train_dataset = DatasetGlob(train_dataset_glob, transform=transformer_dist)
# plt.imshow(train_dataset[0][0], cmap="gray")
# plt.show()


# img_squeeze = train_dataset[0].unsqueeze(0)
# %%


# def my_collate(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch)

transform = transforms.Compose(
    [transform_mask_to_dist, transforms.ToTensor()]
)

dataloader = DataModule(
    datasets.ImageFolder(train_data_path, transform=transform),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    # transform=transform,
)

# dataloader = DataLoader(train_dataset, batch_size=batch_size,
#                         shuffle=True, num_workers=2**4, pin_memory=True, collate_fn=my_collate)
import bio_vae

model = bio_vae.models.create_model("resnet50_vqvae_legacy", **vars(args))
from bio_vae import shapes

lit_model = shapes.MaskEmbed(model,args)
# model = Mask_VAE("VAE", 1, 64,
#                      #  hidden_dims=[32, 64],
#                      image_dims=(interp_size, interp_size))

# model = Mask_VAE(VAE())
# %%
# lit_model = LitAutoEncoderTorch(model)

dataloader.setup()
model.eval()
# %%


model_name = model._get_name()
model_dir = f"my_models/{dataset}_{model_name}"

tb_logger = pl_loggers.TensorBoardLogger(f"logs/")

Path(f"{model_dir}/").mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(dirpath=f"{model_dir}/", save_last=True)

trainer = pl.Trainer(
    logger=tb_logger,
    gradient_clip_val=0.5,
    enable_checkpointing=True,
    devices="auto",
    accelerator="gpu",
    accumulate_grad_batches=4,
    callbacks=[checkpoint_callback],
    min_epochs=50,
    max_epochs=args.epochs,
)  # .from_argparse_args(args)

# %%

try:
    trainer.fit(lit_model, datamodule=dataloader, ckpt_path=f"{model_dir}/last.ckpt")
except:
    trainer.fit(lit_model, datamodule=dataloader)

model.eval()

validation = trainer.validate(lit_model, datamodule=dataloader)
# testing = trainer.test(lit_model, datamodule=dataloader)
example_input = Variable(torch.rand(1, *args.input_dim))

torch.jit.save(model.to_torchscript(), "model.pt")
torch.onnx.export(model, example_input, f"{model_dir}/model.onnx")

# %%
# Inference

