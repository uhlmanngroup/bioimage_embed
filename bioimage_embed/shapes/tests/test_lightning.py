from bioimage_embed.shapes.lightning import MaskEmbed, MaskEmbedSupervised
import pytest
from bioimage_embed import create_model
from torchvision.datasets import FakeData
import pytorch_lightning as pl
from bioimage_embed.lightning.dataloader import DataModule
from torchvision.transforms import transforms
from types import SimpleNamespace


@pytest.fixture
def transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )


@pytest.fixture
def dataset(transform):
    return FakeData(
        size=64,
        image_size=(3, 224, 224),
        num_classes=2,
        transform=transform,
    )


@pytest.fixture
def model():
    return create_model(
        model="resnet18_vae",
        input_dim=[3, 224, 224],
        latent_dim=64,
        pretrained=True,
    )


# TODO Add tests for MaskEmbedSupervised
@pytest.fixture(params=[MaskEmbed, MaskEmbedSupervised])
def wrapper(request):
    return request.param


@pytest.fixture
def lit_model(model, wrapper):
    args = SimpleNamespace(frobenius_norm=False)
    return wrapper(model, args)


@pytest.fixture
def trainer():
    return pl.Trainer(
        max_epochs=1,
        max_steps=1,
        # gpus=1,
        fast_dev_run=True,
    )


@pytest.fixture
def dataloader(dataset):
    return DataModule(
        dataset,
        batch_size=16,
        num_workers=0,
    )


def test_model(trainer, lit_model, dataloader):
    return trainer.test(lit_model, dataloader)


def test_model_fit(trainer, lit_model, dataloader):
    return trainer.fit(lit_model, dataloader)
