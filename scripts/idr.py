# %%
# Import necessary libraries
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
# This allows for caching results of functions to avoid reprocessing the same data.
memory = Memory(location=".", verbose=0)

# %%
# Define FTP host, root directory, and dataset information
# We will download images from an FTP server to use in training our model.
host = "ftp.ebi.ac.uk"
root = "pub/databases/IDR"
dataset = "idr0093-mueller-perturbation"

# Create a glob pattern to match .tif and .tiff files in the dataset directory.
glob_str = f"{root}/{dataset}/**/*.tif*"

# Define the number of GPUs per node and the number of nodes for distributed training.
NUM_GPUS_PER_NODE = 1
NUM_NODES = 1

# %%
# Setup fsspec filesystem for FTP access
# fsspec is used to interact with remote filesystems. Here, we set it up for FTP.
fs = fsspec.filesystem("ftp", host=host, anon=True)


# %%
# Function to get the list of files matching the glob pattern
# This function uses joblib's memory caching to avoid re-fetching the file list.
@memory.cache
def get_file_list(glob_str, fs):
    return fs.glob(glob_str, recursive=True)


# %%
# Function to read a file from the FTP server
def read_file(x):
    try:
        # Attempt to open the image file and read its contents.
        print(x[0])
        stream = x[1].read()
        return stream
    except Exception:
        # If an error occurs (e.g., the file is not found), return None.
        return None


# %%
# Function to convert the binary stream into a PIL Image object
def read_image(x):
    return Image.open(io.BytesIO(x))


# %%
# Function to check if the image is valid
def is_valid_image(x):
    try:
        # Try to open the image and verify that it's not corrupted.
        image = read_image(x)
        image.verify()
        return True
    except (IOError, UnidentifiedImageError):
        # If the image is invalid or corrupted, return False.
        return False


# %%
# Function to convert the image to RGB format
# This ensures that the image has 3 channels (Red, Green, Blue).
def image_open(x):
    return Image.open(io.BytesIO(x)).convert("RGB")


# %%
# Function to add a label to the image
# For this example, we are just adding a label of 0 to each image.
def add_label(x):
    return x, 0


# %%
# The main training function
def train(num_gpus_per_node=1, num_nodes=1):
    # Setup fsspec filesystem for FTP access
    fs = fsspec.filesystem("ftp", host=host, anon=True)

    # Get the list of files to process
    files = get_file_list(glob_str, fs)

    # Instantiate the data transformation pipeline from the configuration
    transform = instantiate(config.Transform())

    # Create the data pipeline using TorchData
    datapipe = (
        IterableWrapper(files)
        .open_files_by_fsspec(
            anon=True,
            protocol="ftp",
            host=host,
            mode="rb",
            filecache={"cache_storage": "/tmp/idr"},
        )
        .map(read_file)  # Read the files from the FTP server
        .filter(filter_fn=is_valid_image)  # Filter out invalid images
        .map(image_open)  # Convert the binary streams to RGB images
        .map(transform)  # Apply transformations to the images
        .map(add_label)  # Add labels to the images
        .set_length(len(files))  # Set the length of the data pipeline
    )

    # Print the first item from the pipeline to check if it's working correctly
    a = next(iter(datapipe))
    print(a)

    # Model training parameters
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

    # Log that training is starting
    print("training")

    # Define the input dimensions for the model
    input_dim = [3, 224, 224]

    # Instantiate the data loader for training
    dataloader = DataModule2(datapipe, num_workers=os.cpu_count())

    # Instantiate the model and wrap it in a Lightning model
    model = config.Model(input_dim=input_dim)
    lit_model = config.LightningModel(model=model)

    # Setup model checkpointing to save the best model
    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        filename="best",
        save_top_k=1,
        mode="min",
        save_last=True,
    )

    # Setup Weights & Biases (Wandb) logger for tracking experiments
    wandb = pl_loggers.WandbLogger(project="idr", name="0093", log_model="all")

    # Setup the trainer with distributed training strategy
    trainer = config.Trainer(
        accelerator="auto",
        devices=num_gpus_per_node,
        num_nodes=num_nodes,
        strategy="ddp",  # Use Distributed Data Parallel (DDP) strategy
        callbacks=[checkpoint],
        logger=[wandb],
    )

    # Combine everything into a configuration object
    cfg = config.Config(
        dataloader=dataloader,
        lit_model=lit_model,
        trainer=trainer,
        recipe=config.Recipe(**params),
    )

    # Instantiate the BioImageEmbed class and start training
    bie = bioimage_embed.BioImageEmbed(cfg)
    wandb.watch(bie.icfg.lit_model, log="all")
    bie.train()
    wandb.finish()


# %%
# Main function to submit the training job
def main():
    logdir = "lightning_slurm/"
    os.makedirs(logdir, exist_ok=True)

    # Submitit executor configuration
    executor = submitit.AutoExecutor(folder=logdir)
    executor.update_parameters(
        mem_gb=2 * 32 * 4,  # 2GB per CPU, 32 CPUs per task, 4 tasks per node
        timeout_min=1440 * 2,  # 48 hours
        gpus_per_node=NUM_GPUS_PER_NODE,
        tasks_per_node=1,
        cpus_per_task=8,
        nodes=NUM_NODES,
        slurm_constraint="a100",
    )
    job = executor.submit(train, NUM_GPUS_PER_NODE, NUM_NODES)
    print(job)


# %%
# Entry point for running the training script directly
if __name__ == "__main__":
    train()
