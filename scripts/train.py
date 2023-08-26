# %%
from pathlib import Path

#  %%
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl

# Note - you must have torchvision installed for this example
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from bioimage_embed.lightning import DatamoduleGlob

from bioimage_embed.datasets import DatasetGlob
from bioimage_embed.models import BioimageEmbed
from bioimage_embed.lightning import LitAutoEncoderTorch
import matplotlib.pyplot as plt


max_epochs = 500

window_size = 128 * 2
batch_size = 16
num_training_updates = 15000

num_hiddens = 64
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3
num_workers = 2**4
data_samples = 16  # Set to -1 for all images
model_name = "VQ_VAE"
dataset = "idr0093"
data_dir = "data"
train_dataset_glob = f"{data_dir}/**/*{dataset}*/**/*tif"

# %%

model_dir = f"models/{dataset}_{model_name}"
# %%

# transformer_crop = CropCentroidPipeline(window_size)
# transformer_dist = MaskToDistogramPipeline(window_size, interp_size)
# transformer_coords = DistogramToCoords(window_size)

train_dataset = DatasetGlob(train_dataset_glob, samples=data_samples)
# train_dataset_crop = DatasetGlob(
#     train_dataset_glob, transform=CropCentroidPipeline(window_size))


transform = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine((0, 360)),
        transforms.RandomResizedCrop(size=512),
        # transforms.RandomCrop(size=(512,512)),
        # transforms.GaussianBlur(5),
        transforms.ToTensor(),
        # transforms.Normalize((0.485), (0.229)),
    ]
)


train_dataset = DatasetGlob(train_dataset_glob, transform=transform, samples=data_samples)
# train_dataset = DatasetGlob(train_dataset_glob, transform=transform)

plt.imshow(train_dataset[10][0], cmap="gray")
# plt.show()
# print(train_dataset[0][0])

# img_squeeze = train_dataset[0].unsqueeze(0)
# %%


# def my_collate(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch)

dataloader = DatamoduleGlob(
    train_dataset_glob,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    transform=transform,
    pin_memory=True,
)

# dataloader = DataLoader(train_dataset, batch_size=batch_size,
#                         shuffle=True, num_workers=2**4, pin_memory=True, collate_fn=my_collate)

model = BioimageEmbed("VQ_VAE", channels=1, num_residual_layers=8, num_residual_hiddens=64)

model = BioimageEmbed("VQ_VAE", channels=1)

# model = Mask_VAE("VAE", 1, 64,
#                      #  hidden_dims=[32, 64],
#                      image_dims=(interp_size, interp_size))

# model = Mask_VAE(VAE())
# %%
lit_model = LitAutoEncoderTorch(model)

tb_logger = pl_loggers.TensorBoardLogger(f"{model_dir}/runs/")

Path(f"{model_dir}/").mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(dirpath=f"{model_dir}/", save_last=True)

trainer = pl.Trainer(
    logger=tb_logger,
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

# %%
