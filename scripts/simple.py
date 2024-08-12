# %%
import bioimage_embed
import bioimage_embed.config as config

# Import necessary modules from bioimage_embed and config.
# bioimage_embed is likely a library designed for embedding biological images,
# and config is used to handle configurations.

# %%
from torchvision.datasets import FakeData
from hydra.utils import instantiate

# Import FakeData from torchvision.datasets to create a fake dataset,
# and instantiate from hydra.utils to create instances based on configuration.

# %%
transform = instantiate(config.Transform())

# Instantiate a transformation using the configuration provided.
# This will likely include any data augmentation or preprocessing steps defined in the configuration.

# %%
dataset = FakeData(
    size=64,
    image_size=(3, 224, 224),
    num_classes=10,
    transform=transform,
)

# Create a fake dataset with 64 images of size 224x224x3 (3 channels), and 10 classes.
# This dataset will be used to simulate data for testing purposes. The 'transform' argument applies the
# transformations defined earlier to the dataset.

# NOTE: The 'dataset' must be a PyTorch Dataset object with X (data) and y (labels).
# If using an unsupervised encoder, set the labels (y) to None; the model will ignore them during training.

# dataset=CelebA(download=True, root="/tmp", split="train")

# The commented-out code suggests an alternative to use the CelebA dataset.
# It would download the CelebA dataset and use the training split, storing it in the '/tmp' directory.

# %% [markdown]
#

# %%
cfg = config.Config(dataset=dataset)
cfg.recipe.model = "resnet18_vae"
cfg.recipe.max_epochs = 100
bie = bioimage_embed.BioImageEmbed(cfg)

# Create a configuration object 'cfg' using the config module, and assign the fake dataset to it.
# The model is set to "resnet18_vae" and the maximum number of epochs for training is set to 100.
# Instantiate the BioImageEmbed object 'bie' using the configuration.


# %%
def process():
    bie.check()
    bie.train()
    bie.export()


# Define a process function that performs three steps:
# 1. 'check()' to verify the setup or configuration.
# 2. 'train()' to start training the model.
# 3. 'export()' to export the trained model.

# %%
# This is the entrypoint for the script and very important if cfg.trainer.num_workers > 0
if __name__ == "__main__":
    process()

# This is the entry point for the script. The 'if __name__ == "__main__":' statement ensures that the 'process()'
# function is called only when the script is run directly, not when imported as a module.
# This is crucial if the 'num_workers' parameter is set in cfg.trainer, as it prevents potential issues
# with multiprocessing in PyTorch.
