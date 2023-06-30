# %%
from pathlib import Path
from torch.autograd import Variable
import torch
import bio_vae
# from bio_vae.models import Bio_VAE
import albumentations as A

# import matplotlib.pyplot as plt
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from PIL import Image

# Note - you must have torchvision installed for this example
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from bio_vae.datasets import DatasetGlob
from bio_vae.lightning import DataModule, DataModuleGlob, LitAutoEncoderTorch
from types import SimpleNamespace


dataset = "ivy_gap"
data_dir = "data"
train_dataset_glob = f"{data_dir}/{dataset}/random/*png"

params = {
    "epochs": 500,
    "batch_size": 16,
    "num_workers": 2**4,
    # "window_size": 64*2,
    "input_dim": (3, 64, 64),
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

# input_dim = (params["channels"], params["window_size"], params["window_size"])
args = SimpleNamespace(**params, **optimizer_params, **lr_scheduler_params)

# num_embeddings = 128
# decay = 0.99
# learning_rate = 1e-2

# latent_dim = 64


transform = A.Compose(
    [
        # A.RandomCrop(
        #     height=512,
        #     width=512,
        #     p=1,
        # ),
        # Flip the images horizontally or vertically with a 50% chance
        A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=0.5,
        ),
        # # Rotate the images by a random angle within a specified range
        # A.Rotate(limit=45, p=0.5),
        # # Randomly scale the image intensity to adjust brightness and contrast
        # A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        # # Apply random elastic transformations to the images
        # A.ElasticTransform(
        #     alpha=1,
        #     sigma=50,
        #     alpha_affine=50,
        #     p=0.5,
        # ),
        # # Shift the image channels along the intensity axis
        # # Add a small amount of noise to the images
        A.RandomCrop(
            height=args.input_dim[1],
            width=args.input_dim[2],
            p=1,
        ),
        # # A.ToFloat(max_value=1, p=1.0),
        A.Normalize(mean=0, std=1, p=1.0),
        ToTensorV2(),
    ]
)

# train_dataset = DatasetGlob(train_dataset_glob)
train_dataset = DatasetGlob(train_dataset_glob, transform=transform)

# train_dataset = DatasetGlob(train_dataset_glob )

# plt.imshow(train_dataset[100][0], cmap="gray")
# plt.show()
# plt.close()

# %%
dataloader = DataModule(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    transform=transform,
    pin_memory=True,
    persistent_workers=True,
    over_sampling=1,
)

# model = bio_vae.models.create_model("resnet150_vqvae_legacy",**vars(args))

model = bio_vae.models.create_model("resnet18_vqvae_legacy", **vars(args))

lit_model = LitAutoEncoderTorch(model, args)

# %%

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
testing = trainer.test(lit_model, datamodule=dataloader)
example_input = Variable(torch.rand(1, *args.input_dim))

torch.jit.save(model.to_torchscript(), "model.pt")
torch.onnx.export(model, example_input, f"{model_dir}/model.onnx")
