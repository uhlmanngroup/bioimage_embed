# %%
import fsspec

# %%
# import torchdata.datapipes as dp
from torchdata.datapipes.iter import IterableWrapper
from PIL import Image
import io
from joblib import Memory
from PIL import UnidentifiedImageError
import bioimage_embed.config as config
from hydra.utils import instantiate
import os
from pytorch_lightning import loggers as pl_loggers
from bioimage_embed.lightning.dataloader import DataModule2

# import fsspec
import submitit
import bioimage_embed
from pytorch_lightning.callbacks import ModelCheckpoint

# %%
memory = Memory(location=".", verbose=0)

# %%
# Define FTP host and root directory
host = "ftp.ebi.ac.uk"
root = "pub/databases/IDR"
dataset = "idr0093-mueller-perturbation"

# %%
# # Setup fsspec filesystem for FTP access
# fs = fsspec.filesystem("ftp", host=host, anon=True)
# fs = fsspec.filesystem("ftp", host=host, anon=True)

# %% [markdown]
# # Glob pattern to match the files you're interested in

# %% [markdown]
# glob_str = f"{root}/{dataset}/**/"
# folders = fs.glob(glob_str, recursive=True)
# dp = IterableWrapper(folders).list_files_by_fsspec(
#     anon=True,
#     protocol="ftp",
#     host=host,
#     recursive=True,
#     masks=["*.tif", "*.tiff"],
# )

# %%
glob_str = f"{root}/{dataset}/**/*.tif*"
NUM_GPUS_PER_NODE = 1
NUM_NODES = 1


# %%
@memory.cache
def get_file_list(glob_str, fs):
    return fs.glob(glob_str, recursive=True)


# %%
# files = get_file_list(glob_str, fs)


# %%
def read_file(x):
    try:
        # Attempt to open the image
        print(x[0])
        stream = x[1].read()
        # print("Valid file")
        return stream
    except Exception:
        # print("Invalid file")
        return None


# %%
def read_image(x):
    return Image.open(io.BytesIO(x))


# %%
def is_valid_image(x):
    try:
        # Attempt to open the image
        image = read_image(x)
        image.verify()  # Ensure it's a valid image
        # print("Valid image")
        return True
    except (IOError, UnidentifiedImageError):
        # print("Invalid image")
        return False


def image_open(x):
    return Image.open(io.BytesIO(x)).convert("RGB")


def add_label(x):
    return x, 0


def train(num_gpus_per_node=1, num_nodes=1):
    # Define FTP host and root directory
    host = "ftp.ebi.ac.uk"
    root = "pub/databases/IDR"
    dataset = "idr0093-mueller-perturbation"

    # # Setup fsspec filesystem for FTP access
    # fs = fsspec.filesystem("ftp", host=host, anon=True)
    fs = fsspec.filesystem("ftp", host=host, anon=True)

    glob_str = f"{root}/{dataset}/**/*.tif*"

    files = get_file_list(glob_str, fs)

    transform = instantiate(config.Transform())

    datapipe = (
        # IterableWrapper(files)
        IterableWrapper(files)
        .open_files_by_fsspec(
            anon=True,
            protocol="ftp",
            host=host,
            mode="rb",
            filecache={"cache_storage": "/tmp/idr"},
        )
        # .filter(filter_fn=is_valid_file)
        .map(read_file)
        .filter(filter_fn=is_valid_image)
        .map(image_open)
        .map(transform)
        .map(add_label)
        .set_length(len(files))
        # .sharding_filter(num_shards=num_nodes, shard_id=0)
        # .batch(1)
        # TODO add zip_with_iter() to combine the image and the label
        # .zip_with_iter()
    )

    # # %%
    # dp = (
    #     # IterableWrapper(files)
    #     IterableWrapper(files)
    #     .open_files_by_fsspec(
    #         anon=True,
    #         protocol="ftp",
    #         host=host,
    #         mode="rb",
    #         filecache={"cache_storage": "tmp/idr"},
    #     )
    #     # .filter(filter_fn=is_valid_file)
    #     .map(read_file)
    #     .filter(filter_fn=is_valid_image)
    #     .map(lambda x: Image.open(io.BytesIO(x)))
    # )

    # %%
    a = next(iter(datapipe))
    print(a)

    # dp = Mapper(dp, lambda x: x.read())
    # dp = Mapper(dp, lambda x: Image.open(io.BytesIO(x)))
    # next(iter(dp))
    # files = fs.glob(glob_str,recursive=True)
    # print(files)
    # # Use FSSpecFileLister to list files from the FTP server
    # # lister_dp = FSSpecFileLister(root=f"ftp://{host}",
    # #     anon=True, protocol="ftp", host=host
    # # )

    # lister_dp = FSSpecFileLister(root=files[0],
    #     anon=True, protocol="ftp", host=host
    # )

    # # Open the listed files using FSSpecFileOpener

    # file_opener_dp = FSSpecFileOpener(lister_dp, mode="rb")

    # first_file = [
    #     f"pub/databases/IDR/idr0093-mueller-perturbation/20200728-ftp/001_B02_T0001F001L01A01Z01C01.tif"
    # ]

    # for i, img in enumerate(dp):
    #     print(img)
    # # %% [markdown]
    # # root = "/nfs/ftp/public/databases/IDR/"
    # # ftp = "ftp://ftp.ebi.ac.uk/pub/databases/idr/"

    # import wandb
    # from pytorch_lightning import LightningModule, Trainer
    # import albumentations as A
    # from types import SimpleNamespace
    # from ray import tune
    # import numpy as np
    # from ray.train.torch import TorchTrainer
    # from ray.train import ScalingConfig
    # from hydra.utils import instantiate
    # import os
    # import glob
    # from PIL import Image
    # from typing import List
    # from torch.utils.data import Dataset
    # import torch
    # from joblib import Memory
    # from pydantic.dataclasses import dataclass
    # from pytorch_lightning import loggers as pl_loggers

    # import os
    # import fsspec
    # from torchdata.datapipes.iter import FSSpecFileLister, FSSpecFileOpener
    # import torchdata.datapipes as dp

    # # "https://ftp.ebi.ac.uk/pub/databases/IDR/"
    # host = "ftp.ebi.ac.uk"
    # root = "pub/databases/IDR"
    # dataset = "idr0093-mueller-perturbation"
    # fs = fsspec.filesystem('ftp', host=host, anon=True)
    # glob_str = f"{root}/{dataset}/**/*.tif*"
    # from torchdata.datapipes.iter import FSSpecFileLister
    # # lister = FSSpecFileLister(root=root, fs=fs, masks=glob_str)
    # from torchdata.datapipes.iter import IterableWrapper, Mapper,MapDataPipe

    # # file_paths_dp = IterableWrapper(lister,ftp=)
    # file_opener_dp = FSSpecFileOpener(file_paths_dp,ftp={"host"=host, "anon"=True})

    # dp = IterableWrapper(["ftp://BUCKET_NAME"]).list_files_by_fsspec()

    # files = fs.glob(glob_str,recursive=True)

    # # fs.ls(f"{root}{dataset}")
    # # 56008 files:
    # files

    # # %% [markdown]
    # # dataset = datasets.ImageFolder(transform=transform)

    params = {
        "model": "resnet50_vqvae",
        # "data": "data",
        "opt": "lamb",
        "latent_dim": 224**2 // 4,
        "max_epochs": 1000,
        "max_steps": -1,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        # "sched": "cosine",
        "epochs": 1000,
        "lr": 1e-3,
        "batch_size": 16,
        "sched": "cosine",
    }
    # memory = Memory(location='.', verbose=0)

    # @memory.cache
    # def get_file_list(glob_str,fs):
    #     # return fs.glob(glob_str)
    #     return fs.open(glob_str,filecache={'cache_storage':'tmp/idr'})
    #     # return fsspec.open_files(glob_str, recursive=True)
    #     # return glob.glob(os.path.join(glob_str), recursive=True)

    # # def collate_fn(batch):
    # #     # Filter out None values
    # #     batch = list(filter(lambda x: x[0] is not None, batch))
    # #     return torch.utils.data.dataloader.default_collate(batch)

    # # class GlobDataset(Dataset):
    # #     def __init__(self, glob_str,transform=None,fs=fsspec.filesystem('file')):
    # #         print("Getting file list, this may take a while")
    # #         self.file_list = get_file_list(glob_str,fs)
    # #         print("Done getting file list")
    # #         self.transform = transform

    # #     def __len__(self):
    # #         return len(self.file_list)

    # #     def __getitem__(self, idx):
    # #         if torch.is_tensor(idx):
    # #             idx = idx.tolist()

    # #         img_name = self.file_list[idx]
    # #         obj = fs.open(img_name,filecache={'cache_storage':'tmp/idr'})
    # #         try:
    # #             with obj as f:
    # #                 image = Image.open(f)
    # #             # image = Image.open(img_name)
    # #         except:
    # #             return None,None
    # #         # breakpoint()
    # #         image = np.array(image)
    # #         if self.transform:
    # #             # t = A.Compose([A.ToRGB(),transform, A.RandomCrop(224,224)])
    # #             t = A.Compose([A.ToRGB(),self.transform])
    # #             image = t(image=image)

    # #         # breakpoint()

    # #         return image["image"], 0

    # # root_dir = '/nfs/research/uhlmann/ctr26/idr/idr0093-mueller-perturbation/'
    # # fs = fsspec.filesystem('file')
    # # fs = fsspec.filesystem(
    # #         'ftp', host='ftp.ebi.ac.uk',
    # #         cache_storage='/tmp/files/')
    # # root_dir = '/pub/databases/IDR/idr0093-mueller-perturbation/'

    # # # /nfs/ftp/public/databases/IDR/idr0093-mueller-perturbation/'
    # # # /nfs/ftp/public/databases/IDR/
    print("training")
    input_dim = [3, 224, 224]

    # mock_dataset = config.ImageFolderDataset(
    #     _target_="bioimage_embed.datasets.FakeImageFolder",
    #     image_size=input_dim,
    #     num_classes=1,
    # )

    # transform = instantiate(config.ATransform())
    # dataset = GlobDataset(root_dir+'**/*.tif*',transform,fs=fs)
    # dataset = RandomDataset(32, 64)
    # dataloader = config.DataLoader(
    #     dataset=dataset, num_workers=os.cpu_count(), batch_size=None
    # )
    dataloader = DataModule2(datapipe, num_workers=os.cpu_count())

    # dataloader = config.DataLoader(num_workers=os.cpu_count())

    # assert instantiate(dataloader, batch_size=1)
    # assert dataset[0]

    model = config.Model(input_dim=input_dim)

    lit_model = config.LightningModel(model=model)

    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        filename="best",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    wandb = pl_loggers.WandbLogger(project="idr", name="0093", log_model="all")
    # wandb.watch(lit_model, log="all")
    trainer = config.Trainer(
        accelerator="auto",
        devices=num_gpus_per_node,
        num_nodes=num_nodes,
        strategy="ddp",
        callbacks=[checkpoint],
        # callbacks=[],
        # plugin=[],
        logger=[wandb],
    )

    cfg = config.Config(
        dataloader=dataloader,
        lit_model=lit_model,
        trainer=trainer,
        recipe=config.Recipe(**params),
    )
    # breakpoint()

    bie = bioimage_embed.BioImageEmbed(cfg)
    wandb.watch(bie.icfg.lit_model, log="all")
    # wandb.run.define_metric("mse/val", summary="best")
    # wandb.run.define_metric("loss/val.loss", summary="best")
    # bie.check()
    bie.train()
    wandb.finish()


def main():
    logdir = "lightning_slurm/"
    os.makedirs(logdir, exist_ok=True)

    # executor is the submission interface (logs are dumped in the folder)
    executor = submitit.AutoExecutor(folder=logdir)
    executor.update_parameters(
        mem_gb=2 * 32 * 4,  # 2GB per CPU, 32 CPUs per task, 4 tasks per node
        timeout_min=1440 * 2,  # 48 hours
        # slurm_partition="your_partition_name",  # Replace with your partition name
        gpus_per_node=NUM_GPUS_PER_NODE,
        tasks_per_node=1,
        cpus_per_task=8,
        nodes=NUM_NODES,
        slurm_constraint="a100",
    )
    job = executor.submit(train, NUM_GPUS_PER_NODE, NUM_NODES)
    print(job)


if __name__ == "__main__":
    train()
