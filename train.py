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


from mask_vae.datasets import DSB2018
from mask_vae.transforms import CropCentroidPipeline, DistogramToCoords, DistogramToCoords, MaskToDistogramPipeline
from mask_vae.models import Mask_VAE, VQ_VAE
from mask_vae.lightning import LitAutoEncoder

interp_size = 128*4

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

train_dataset_glob = os.path.join(os.path.expanduser("~"),
                                  "data-science-bowl-2018/stage1_train/*/masks/*.png")
# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")

transformer_crop = CropCentroidPipeline(window_size)
transformer_dist = MaskToDistogramPipeline(window_size, interp_size)
transformer_coords = DistogramToCoords(window_size)

# train_dataset_raw = DSB2018(train_dataset_glob)
# train_dataset_crop = DSB2018(
#     train_dataset_glob, transform=CropCentroidPipeline(window_size))
train_dataset = DSB2018(train_dataset_glob, transform=transformer_dist)

# img_squeeze = train_dataset_crop[0].unsqueeze(0)
# img_crop = train_dataset_crop[0]


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8, pin_memory=True, collate_fn=my_collate)

model = Mask_VAE(VQ_VAE(channels=1))
lit_model = LitAutoEncoder(model)

tb_logger = pl_loggers.TensorBoardLogger("runs/")

Path("checkpoints/").mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints/",
    save_last=True
)

trainer = pl.Trainer(
    logger=tb_logger,
    enable_checkpointing=True,
    gpus=1,
    accumulate_grad_batches=1,
    callbacks=[checkpoint_callback],
    min_epochs=50,
    max_epochs=75,
)  # .from_argparse_args(args)


try:
    trainer.fit(lit_model, dataloader,
                ckpt_path="checkpoints/last.ckpt")
except:
    trainer.fit(lit_model, dataloader)
