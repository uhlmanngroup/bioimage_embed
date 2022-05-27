# %%
import os
from pathlib import Path

#  %%
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch
# Note - you must have torchvision installed for this example
from torch.utils.data import DataLoader
import os
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms


from mask_vae.datasets import DSB2018
from mask_vae.transforms import CropCentroidPipeline, DistogramToCoords, DistogramToCoords, MaskToDistogramPipeline
from mask_vae.models import Mask_VAE, VQ_VAE, VAE
from mask_vae.lightning import LitAutoEncoderTorch, LitAutoEncoderPyro
import matplotlib.pyplot as plt

interp_size = 128*4

max_epochs = 5000

window_size = 128*4
batch_size = 2
num_training_updates = 15000

num_hiddens = 64
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-4

# train_dataset_glob = "data-science-bowl-2018/stage1_train/*/masks/*.png"
train_dataset_glob = "data/stage1_train/*/masks/*.png"

# train_dataset_glob = os.path.join("data/BBBC010_v1_foreground_eachworm/*.png")


# train_dataset_glob = os.path.join("data/dsb2018/train/masks/*.tif")
# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")

# model_dir = "test"
# model_dir = "BBBC010_v1_foreground_eachworm"
model_dir = "dsb2018"
# %%

transformer_crop = CropCentroidPipeline(window_size)
transformer_dist = MaskToDistogramPipeline(window_size, interp_size)
transformer_coords = DistogramToCoords(window_size)

train_dataset = DSB2018(train_dataset_glob)
# train_dataset_crop = DSB2018(
#     train_dataset_glob, transform=CropCentroidPipeline(window_size))


transform = transforms.Compose(
    [
        transformer_crop,
        transformer_dist,

    ]
)

train_dataset = DSB2018(train_dataset_glob, transform=transformer_dist)
plt.imshow(train_dataset[0][0],cmap='gray')
plt.show()

train_dataset = DSB2018(train_dataset_glob, transform=transforms.ToTensor())
plt.imshow(train_dataset[0][0],cmap='gray')
plt.show()
train_dataset = DSB2018(train_dataset_glob, transform=transformer_crop)
plt.imshow(train_dataset[0][0],cmap='gray')
plt.show()


train_dataset = DSB2018(train_dataset_glob, transform=transformer_crop)
plt.imshow(train_dataset[0][0],cmap='gray')
plt.show()

# img_squeeze = train_dataset[0].unsqueeze(0)
# %%


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=2**8, pin_memory=True, collate_fn=my_collate)

model = Mask_VAE(VQ_VAE(channels=1))
model = Mask_VAE(VAE(1, 64,
                     #  hidden_dims=[32, 64],
                     image_dims=(interp_size, interp_size)))

# model = Mask_VAE(VAE())

lit_model = LitAutoEncoderTorch(model)

tb_logger = pl_loggers.TensorBoardLogger("runs/")

Path(f'{model_dir}/').mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=f'{model_dir}/',
    save_last=True
)

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
    trainer.fit(lit_model, dataloader,
                ckpt_path=f'{model_dir}/last.ckpt')
except:
    trainer.fit(lit_model, dataloader)

# %%
