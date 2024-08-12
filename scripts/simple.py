# %%
import bioimage_embed
import bioimage_embed.config as config

# %%
from torchvision.datasets import FakeData
from hydra.utils import instantiate


# %%
transform = instantiate(config.Transform())

# # %%
dataset = FakeData(
    size=64,
    image_size=(3, 224, 224),
    num_classes=10,
    transform=transform,
)
# dataset=CelebA(download=True, root="/tmp", split="train")

# %% [markdown]

# %%
cfg = config.Config(dataset=dataset)
cfg.recipe.model = "resnet18_vae"
cfg.recipe.max_epochs = 100
bie = bioimage_embed.BioImageEmbed(cfg)


# %%
def process():
    bie.check()
    bie.train()
    bie.export()


# %%
# This is the entrypoint for the script and very import if cfg.trainer.num_workers > 0
if __name__ == "__main__":
    process()
