# %%
import fsspec
from torchdata.datapipes.iter import IterableWrapper
from PIL import Image, UnidentifiedImageError
import io
from joblib import Memory
from pytorch_lightning import loggers as pl_loggers
from bioimage_embed.lightning.dataloader import DataModule2
import submitit
import bioimage_embed
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from hydra.utils import instantiate
import bioimage_embed.config as config

# %%
# Setup joblib memory caching
memory = Memory(location=".", verbose=0)

# %%
# Define FTP host and root directory
host = "ftp.ebi.ac.uk"
root = "pub/databases/IDR"
dataset = "idr0093-mueller-perturbation"
glob_str = f"{root}/{dataset}/**/*.tif*"
NUM_GPUS_PER_NODE = 1
NUM_NODES = 1

# %%
# Setup fsspec filesystem for FTP access
fs = fsspec.filesystem("ftp", host=host, anon=True)


# %%
@memory.cache
def get_file_list(glob_str, fs):
    return fs.glob(glob_str, recursive=True)


# %%
def read_file(x):
    try:
        # Attempt to open the image
        print(x[0])
        stream = x[1].read()
        return stream
    except Exception:
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
        return True
    except (IOError, UnidentifiedImageError):
        return False


# %%
def image_open(x):
    return Image.open(io.BytesIO(x)).convert("RGB")


# %%
def add_label(x):
    return x, 0


# %%
def train(num_gpus_per_node=1, num_nodes=1):
    # Setup fsspec filesystem for FTP access
    fs = fsspec.filesystem("ftp", host=host, anon=True)

    files = get_file_list(glob_str, fs)
    transform = instantiate(config.Transform())

    datapipe = (
        IterableWrapper(files)
        .open_files_by_fsspec(
            anon=True,
            protocol="ftp",
            host=host,
            mode="rb",
            filecache={"cache_storage": "/tmp/idr"},
        )
        .map(read_file)
        .filter(filter_fn=is_valid_image)
        .map(image_open)
        .map(transform)
        .map(add_label)
        .set_length(len(files))
    )

    a = next(iter(datapipe))
    print(a)

    params = {
        "model": "resnet50_vqvae",
        "opt": "lamb",
        "latent_dim": 224**2 // 4,
        "max_epochs": 1000,
        "max_steps": -1,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "sched": "cosine",
        "epochs": 1000,
        "lr": 1e-3,
        "batch_size": 16,
    }

    print("training")
    input_dim = [3, 224, 224]

    dataloader = DataModule2(datapipe, num_workers=os.cpu_count())

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
    trainer = config.Trainer(
        accelerator="auto",
        devices=num_gpus_per_node,
        num_nodes=num_nodes,
        strategy="ddp",
        callbacks=[checkpoint],
        logger=[wandb],
    )

    cfg = config.Config(
        dataloader=dataloader,
        lit_model=lit_model,
        trainer=trainer,
        recipe=config.Recipe(**params),
    )

    bie = bioimage_embed.BioImageEmbed(cfg)
    wandb.watch(bie.icfg.lit_model, log="all")
    bie.train()
    wandb.finish()


# %%
def main():
    logdir = "lightning_slurm/"
    os.makedirs(logdir, exist_ok=True)

    # Submitit executor configuration
    executor = submitit.AutoExecutor(folder=logdir)
    executor.update_parameters(
        mem_gb=2 * 32 * 4,
        timeout_min=1440 * 2,
        gpus_per_node=NUM_GPUS_PER_NODE,
        tasks_per_node=1,
        cpus_per_task=8,
        nodes=NUM_NODES,
        slurm_constraint="a100",
    )
    job = executor.submit(train, NUM_GPUS_PER_NODE, NUM_NODES)
    print(job)


# %%
if __name__ == "__main__":
    train()
