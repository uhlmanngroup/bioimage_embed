# %%
# Import necessary modules
import bioimage_embed
import bioimage_embed.config as config
from hydra.utils import instantiate
from torchvision import datasets

# %%
# Define input dimensions
input_dim = [3, 224, 224]

# %%
# Use the default augmentation list
transform = instantiate(config.Transform())
transform.transform

# %%
# Load the CelebA dataset with the specified transformations
dataset = datasets.CelebA(
    root="data/",
    split="train",
    target_type="attr",
    download=True,
    transform=transform,
)

# %%
# Create a dataloader from the dataset
dataloader = config.DataLoader(dataset=dataset)

# %%
# Instantiate the model with the input dimensions
model = config.Model(input_dim=input_dim)

# %%
# Define the recipe for the model
recipe = config.Recipe(model="resnet18_vae")

# %%
# Create the configuration object with the recipe, dataloader, and model
cfg = config.Config(recipe=recipe, dataloader=dataloader, model=model)

# %%
# Initialize BioImageEmbed with the configuration
bie = bioimage_embed.BioImageEmbed(cfg)

# %%
# Train and export the model if this script is run as the main program
if __name__ == "__main__":
    bie.check().train().export("model")
# lit_model = bie.check().train().get_model()
# bie.export("model")
