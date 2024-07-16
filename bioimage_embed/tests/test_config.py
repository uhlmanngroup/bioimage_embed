from .. import config
from .. import bioimage_embed
import pytest
from hydra.utils import instantiate
from torchvision.datasets import FakeData
from torchvision import transforms
from omegaconf import OmegaConf
schema_map = config.__schemas__
schemas = list(schema_map.values())


@pytest.mark.parametrize("Schema", schemas)
def test_schema(Schema):
    Schema()


@pytest.mark.parametrize("Schema", schemas)
def test_instantiate(Schema):
    schema = config.resolve_schema(Schema())
    obj = instantiate(schema)
    assert obj is not None, "obj should not be None"

@pytest.fixture
def input_dim(): 
    return [3, 224, 224]

@pytest.fixture
def dataset(input_dim):
    transform = transforms.ToTensor()
    return FakeData(size=64,
                       image_size=input_dim,
                       num_classes=1,
                       transform=transform)
@pytest.fixture
def dataloader(dataset):
    return config.DataLoader(dataset=dataset, num_workers=0)

@pytest.fixture
def model(input_dim):
    return config.Model(input_dim=input_dim)

@pytest.fixture
def cfg(dataloader,model):
    return config.Config(dataloader=dataloader, model=model)


def test_config(cfg):
    assert cfg is not None, "Config should not be None"


def test_config_instantiate(cfg):
    assert instantiate(cfg) is not None, "Config should not be None"

def test_resolve(bie):
    assert bie.resolve()

@pytest.fixture
def bie(cfg):
    return bioimage_embed.BioImageEmbed(cfg)


def test_model_check(bie):
    bie.model_check()


def test_train_check(bie):
    bie.trainer_check()
