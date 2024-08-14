import pytest
import torch
import pytorch_lightning as pl
from bioimage_embed.models import __all_models__
from bioimage_embed.lightning import (
    DataModule,
    AESupervised,
    AEUnsupervised,
)
from bioimage_embed.models import create_model
from bioimage_embed import config
from hydra.utils import instantiate
from torchvision.datasets import FakeData

torch.manual_seed(42)


@pytest.fixture()
def transform():
    return instantiate(config.Transform())


@pytest.fixture(params=[1, 2, 16])
def classes(request):
    return request.param


@pytest.fixture(params=__all_models__)
def model_name(request):
    return request.param


@pytest.fixture()
def image_dim():
    return (224, 224)


@pytest.fixture()
def channel_dim():
    return 3


@pytest.fixture()
def samples():
    return 32


@pytest.fixture(params=[16])
def latent_dim(request):
    return request.param


@pytest.fixture(
    params=[
        4,
    ]
)
def batch_size(request):
    return request.param


@pytest.fixture()
def pretrained():
    return True


@pytest.fixture()
def progress():
    return True


# TODO put this in a conftest.py file
@pytest.fixture
def model(model_name, image_dim, channel_dim, latent_dim, pretrained, progress):
    input_dim = (channel_dim, *image_dim)
    return create_model(
        model_name,
        input_dim,
        latent_dim,
        pretrained,
        progress,
    )


@pytest.fixture()
def dummy_model(channel_dim, image_dim, latent_dim):
    return create_model(
        "dummy_model",
        input_dim=(channel_dim, *image_dim),
        latent_dim=latent_dim,
        pretrained=False,
        progress=False,
    )


@pytest.fixture()
def input_dim(image_dim, channel_dim):
    return (channel_dim, *image_dim)


@pytest.fixture()
def data(input_dim):
    return torch.rand(*input_dim)


@pytest.fixture()
def dataset(samples, input_dim, transform, classes=2):
    # x = torch.rand(samples, *input_dim)
    # y = torch.torch.randint(classes - 1, (samples,))
    # return TensorDataset(x, y)
    return FakeData(
        size=samples,
        image_size=input_dim,
        num_classes=classes,
        transform=transform,
    )


@pytest.fixture(params=[AESupervised, AEUnsupervised])
def lit_model_wrapper(request):
    return request.param


# @pytest.mark.skip(reason="Dictionaries not allowed")
# def test_export_onxx(data, lit_model):
#     return lit_model.to_onnx("model.onnx", data)


@pytest.fixture()
def datamodule(dataset, batch_size):
    return DataModule(
        dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=0,  # This avoids processes being forked
        pin_memory=False,
    )


@pytest.fixture()
def trainer():
    return pl.Trainer(
        # max_steps=1,
        max_epochs=2,
    )


@pytest.fixture()
def model_torchscript(lit_model):
    return lit_model.to_torchscript()


@pytest.fixture()
def lit_dummy_model(lit_model_wrapper, dummy_model):
    return lit_model_wrapper(dummy_model)


@pytest.fixture()
def lit_model(lit_model_wrapper, model):
    return lit_model_wrapper(model)


def test_trainer_test(trainer, lit_model, datamodule):
    return trainer.test(lit_model, datamodule)


def test_trainer_dummy_model_fit(trainer, lit_dummy_model, datamodule):
    return trainer.fit(lit_dummy_model, datamodule)


# @pytest.mark.skip(reason="Expensive")
def test_trainer_fit(trainer, lit_model, datamodule):
    return trainer.fit(lit_model, datamodule)


@pytest.mark.skip(reason="needs batched data")
def test_dataset_trainer(trainer, lit_model, dataset):
    return trainer.test(lit_model, dataset)


def test_model_properties(model):
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.latent_dim is not None
    assert model.input_dim is not None
    assert model.model_name is not None
    assert model.model_config is not None


def test_trainer_predict(trainer, lit_model, datamodule):
    batch_size = datamodule.predict_dataloader().batch_size
    latent_dim = lit_model.model.latent_dim
    predictions = trainer.predict(lit_model, datamodule)
    assert predictions is not None
    assert len(predictions[0].z.flatten()) == batch_size * latent_dim
    # TODO prefer
    # assert list(predictions[0].z.shape) == [batch_size,latent_dim]
    # assert len(list(predictions[0].z.shape)) == 2


# Has to be a list not a tuple
def test_export_onnx(lit_model, data):
    example_input = data.unsqueeze(0)
    return lit_model.to_onnx("model.onnx", example_input, export_params=True)


@pytest.mark.skip(reason="models cant take in variable length args and kwargs")
def test_export_jit(model_torchscript):
    return model_torchscript.save("model.pt")


@pytest.mark.skip(reason="models cant take in variable length args and kwargs")
def test_jit_save(model_torchscript):
    return torch.jit.save(model_torchscript, "model.pt", method="script")
