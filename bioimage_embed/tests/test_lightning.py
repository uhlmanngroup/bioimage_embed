import pytest
import torch
import pytorch_lightning as pl
from bioimage_embed.models import MODELS
from bioimage_embed.lightning import (
    DataModule,
    AutoEncoderSupervised,
    AutoEncoderUnsupervised,
)
from bioimage_embed.models import create_model
from torch.utils.data import TensorDataset


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
def latent_dim():
    return 16


@pytest.fixture(params=[1, 2, 16])
def batch_size(request):
    return request.param


@pytest.fixture()
def pretrained():
    return True


@pytest.fixture()
def progress():
    return True


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
def batched_data(data):
    return data.unsqueeze(0)


@pytest.fixture()
def batched_dataset(batched_data):
    return TensorDataset(batched_data, torch.zeros(1))


@pytest.fixture()
def dataset(data):
    return TensorDataset(data, torch.zeros(1))


# @pytest.fixture()
# def unlabelled_dataset(data):
#     return data


# @pytest.fixture()
# def supervised_lit_model(model):
#     return AutoEncoderSupervised(model)


# @pytest.fixture(params=[AutoEncoderSupervised, AutoEncoderUnsupervised])
def lit_model(lit_model, model):
    return AutoEncoderSupervised(model)


@pytest.fixture(
    params=[AutoEncoderSupervised, AutoEncoderUnsupervised]
)
def lit_model(request, model):
    return request.param(model)


# @pytest.fixture()
# def unsupervised_lit_model(model):
#     return AutoEncoderUnsupervised(model)


@pytest.fixture()
def labelled_data(data):
    return data, torch.tensor([0])


# @pytest.mark.skip(reason="Dictionaries not allowed")
# def test_export_onxx(data, lit_model):
#     return lit_model.to_onnx("model.onnx", data)


@pytest.fixture()
def data(input_dim):
    return torch.rand(1, *input_dim)


@pytest.fixture()
def dataloader(dataset, batch_size):
    return DataModule(
        dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=1,
        pin_memory=False,
    )


# @pytest.fixture()
# def unlabelled_dataloader(unlabelled_dataset, batch_size):
#     return DataModule(
#         unlabelled_dataset,
#         batch_size=batch_size,
#         # shuffle=True,
#         num_workers=1,
#         pin_memory=False,
#     )


@pytest.fixture()
def dataloader(dataset, batch_size):
    return DataModule(
        dataset,
        batch_size=batch_size,
        # shuffle=True,
        num_workers=1,
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


def test_dataset_trainer(trainer, lit_model, dataset):
    return trainer.test(lit_model, dataset)


@pytest.mark.skip(reason="Dictionaries not allowed, pythae uses dictionaries")
def test_export_onnx(lit_model):
    return lit_model.to_onnx("model.onnx",export_params=True)


@pytest.mark.skip(reason="Upstream bug with pythae")
def test_export_jit(model_torchscript):
    return model_torchscript.save("model.pt")


@pytest.mark.skip(reason="Upstream bug with pythae")
def test_jit_save(model_torchscript):
    return torch.jit.save(model_torchscript, "model.pt", method="script")
