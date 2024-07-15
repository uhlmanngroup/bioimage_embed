import pytest
import torch
import pytorch_lightning as pl
from bioimage_embed.models import MODELS
from bioimage_embed.lightning import (
    DataModule,
    AESupervised,
    AEUnsupervised,
)
from bioimage_embed.models import create_model
from torch.utils.data import TensorDataset


torch.manual_seed(42)

@pytest.fixture(params=MODELS)
def model_name(request):
    return request.param


@pytest.fixture()
def image_dim():
    return (256, 256)


@pytest.fixture()
def channel_dim():
    return 3

@pytest.fixture()
def samples():
    return 32

@pytest.fixture(params=[16])
def latent_dim(request):
    return request.param


@pytest.fixture(params=[4,])
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
def input_dim(image_dim, channel_dim):
    return (channel_dim, *image_dim)


@pytest.fixture()
def data(input_dim):
    return torch.rand(*input_dim)

@pytest.fixture()
def dataset(samples, input_dim,classes=2):
    x = torch.rand(samples, *input_dim)
    y = torch.torch.randint(classes-1,(samples,))
    return TensorDataset(x,y)



@pytest.fixture(
    params=[AESupervised, AEUnsupervised]
)
def lit_model(request, model):
    return request.param(model)

# @pytest.mark.skip(reason="Dictionaries not allowed")
# def test_export_onxx(data, lit_model):
#     return lit_model.to_onnx("model.onnx", data)


@pytest.fixture()
def dataloader(dataset, batch_size):
    return DataModule(
        dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=0, # This avoids processes being forked
        pin_memory=False,
    )


@pytest.fixture()
def trainer():
    return pl.Trainer(
        max_steps=1,
        max_epochs=1,
    )

@pytest.fixture()
def model_torchscript(lit_model):
    return lit_model.to_torchscript()

def test_trainer_test(trainer, lit_model, dataloader):
    return trainer.test(lit_model, dataloader)


@pytest.mark.skip(reason="Expensive")
def test_trainer_fit(trainer, lit_model, dataloader):
    return trainer.fit(lit_model, dataloader)

@pytest.mark.skip(reason="needs batched data")
def test_dataset_trainer(trainer, lit_model, dataset):
    return trainer.test(lit_model, dataset)


# Has to be a list not a tuple
def test_export_onnx(lit_model, data):
    example_input = data.unsqueeze(0)
    return lit_model.to_onnx("model.onnx",example_input,export_params=True)


@pytest.mark.skip(reason="models cant take in variable length args and kwargs")
def test_export_jit(model_torchscript):
    return model_torchscript.save("model.pt")


@pytest.mark.skip(reason="models cant take in variable length args and kwargs")
def test_jit_save(model_torchscript):
    return torch.jit.save(model_torchscript, "model.pt", method="script")
