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
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from bioimage_embed.lightning.dataloader import StratifiedSampler


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


@pytest.mark.skip(reason="Expensive")
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


@pytest.fixture(params=[128])
def large_batch(request):
    return request.param


@pytest.fixture
def large_data(input_dim, large_batch):
    return torch.empty(large_batch**2, *input_dim)


@pytest.fixture(params=[8])
def many_classes(request):
    return request.param


@pytest.fixture
def imbalanced_dataset(large_data, many_classes):
    """
    Return 'classes' and an imbalanced distribution 'p'. 'classes' can be any length.
    The distribution 'p' must sum to 1.
    """
    data, classes = large_data, many_classes
    samples = len(data)
    if classes == 1:
        pytest.skip("Cannot create an imbalanced dataset with only one class.")

    p = 2 ** np.arange(1, classes + 1)

    p = p / p.sum()  # Normalize to sum to 1

    labels = np.random.choice(a=np.arange(classes), size=(samples,), p=p)
    # Set the dataset's targets
    # dataset.targets = torch.tensor(labels)
    dataset = TensorDataset(data, torch.tensor(labels))
    return dataset


@pytest.fixture(params=[16])
def batch_split(request):
    return request.param


@pytest.fixture()
def stratified_dataloader(imbalanced_dataset):
    dataset = imbalanced_dataset
    samples = len(dataset)

    return DataLoader(
        dataset,
        batch_size=int(np.sqrt(samples)),
        sampler=StratifiedSampler(dataset),
        num_workers=0,
        drop_last=True,
    )


def test_stratified_sampler(stratified_dataloader):
    # Unpack the dataloader
    dataloader = stratified_dataloader

    # Collect all sampled labels
    all_labels = []
    for inputs, labels in dataloader:
        all_labels.extend(labels.numpy())

    # Convert to NumPy array
    all_labels = np.array(all_labels)

    # Calculate the number of classes
    num_classes = len(np.unique(all_labels))

    # Calculate the sampled label distribution
    sampled_counts = np.bincount(all_labels, minlength=num_classes)
    sampled_distribution = sampled_counts / len(all_labels)

    # Expected proportion for each class (uniform distribution)
    expected_proportion = num_classes * [1.0 / num_classes]
    # Assert that the sampled distribution is close to the expected proportions
    assert np.allclose(
        sampled_distribution, expected_proportion, atol=0.05
    ), f"Sampled distribution {sampled_distribution} does not match expected {expected_proportion}"


def test_sanity_check_stratified():
    # Create an imbalanced dataset (e.g., class 0 has more samples than class 1)
    labels = [0] * 80 + [1] * 20
    dataset = [(data, label) for data, label in zip(range(100), labels)]

    # Initialize the sampler
    sampler = StratifiedSampler(dataset)

    # Create a DataLoader
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    # Collect sampled labels
    sampled_labels = []
    for _, label_batch in dataloader:
        sampled_labels.extend(label_batch.numpy())

    # Analyze the distribution
    sampled_labels = np.array(sampled_labels)
    sampled_counts = np.bincount(sampled_labels)
    sampled_distribution = sampled_counts / len(sampled_labels)

    print("Sampled Label Distribution:")
    for i, proportion in enumerate(sampled_distribution):
        print(f"Class {i}: {proportion*100:.2f}%")
