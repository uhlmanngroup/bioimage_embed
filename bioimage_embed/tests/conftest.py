# from bioimage_embed import config
from torchvision.datasets import FakeData
import pytest
from torchvision import transforms

# from bioimage_embed.bie import BioImageEmbed
from .. import config
from ..bie import BioImageEmbed

from hydra import initialize, compose


@pytest.fixture
def input_dim():
    return [3, 224, 224]


@pytest.fixture
def dataset(input_dim):
    transform = transforms.ToTensor()
    return FakeData(size=64, image_size=input_dim, num_classes=2, transform=transform)


@pytest.fixture
def lite_model():
    return "dummy_model"


@pytest.fixture
def bie(cfg):
    return BioImageEmbed(cfg)


@pytest.fixture
def hydra_cfg():
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        return cfg


@pytest.fixture
def cfg_recipe(lite_model):
    return config.Recipe(model=lite_model)


@pytest.fixture
def cfg_trainer():
    return config.Trainer(max_epochs=1, max_steps=1, fast_dev_run=True)


@pytest.fixture
def cfg_dataloader(dataset):
    return config.DataLoader(dataset=dataset, num_workers=0)


# TODO double check this is sensible
@pytest.fixture
def cfg(cfg_recipe, cfg_trainer, cfg_dataloader):
    cfg = config.Config(
        recipe=cfg_recipe, trainer=cfg_trainer, dataloader=cfg_dataloader
    )
    return cfg
    # This is an alternative way to create a config object but it is less flexible and if the config object is changed in the future, this will break, i.e validation is not guaranteed

    # cfg.dataloader.num_workers = 0  # This avoids processes being forked
    # cfg.trainer.max_epochs = 1
    # cfg.trainer.max_steps = 1
    # cfg.trainer.fast_dev_run = True
    # cfg.recipe.model = model
