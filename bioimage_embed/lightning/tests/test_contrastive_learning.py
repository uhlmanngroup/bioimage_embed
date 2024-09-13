import torch
import pytest
from ..torch import create_label_based_pairs, compute_contrastive_loss

torch.manual_seed(42)


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def latent_dim():
    return 16


@pytest.fixture(params=[1, 2, 4])
def classes(request):
    return request.param


@pytest.fixture()
def labels(batch_size, classes):
    return torch.randint(0, classes + 1, (batch_size, 1))


@pytest.fixture
def features(batch_size, latent_dim):
    return torch.rand(batch_size, latent_dim)


def test_create_label_based_pairs_single_label(features, batch_size):
    # All labels are the same
    labels = torch.zeros(batch_size, 1)
    input_pairs, target_pairs = create_label_based_pairs(features, labels)
    assert (
        input_pairs.numel() == 0 and target_pairs.numel() == 0
    ), "Should return empty tensors for single label"


def test_create_label_based_pairs_no_pair(features, labels):
    # Case when each label only has one sample
    input_pairs, target_pairs = create_label_based_pairs(features, labels)
    if torch.unique(labels).size(0) == labels.size(0):
        assert (
            input_pairs.numel() == 0 and target_pairs.numel() == 0
        ), "Should return empty tensors when no pairs are available"


def test_create_label_based_pairs_multiple_pairs(features, labels):
    # Check if multiple pairs can be created
    input_pairs, target_pairs = create_label_based_pairs(features, labels)
    if torch.unique(labels).size(0) > 1:
        assert input_pairs.size(0) == target_pairs.size(
            0
        ), "Input and target pairs should have the same number of pairs"
        assert input_pairs.size(1) == features.size(
            1
        ), "Pair feature size should match the input feature size"


def test_compute_contrastive_loss_single_label(features, batch_size):
    # All labels are the same
    labels = torch.zeros(batch_size, 1)
    loss = compute_contrastive_loss(features, labels)
    assert (
        loss.item() == 0.0
    ), "Loss should be zero when all labels are the same (no valid pairs)"


def test_compute_contrastive_loss_no_valid_pairs(features, labels):
    # Case when each label only has one sample
    if torch.unique(labels).size(0) == labels.size(0):  # All labels are unique
        loss = compute_contrastive_loss(features, labels)
        assert (
            loss.item() == 0.0
        ), "Loss should be zero when no valid pairs are available"


def test_compute_contrastive_loss_valid_pairs(features, labels):
    # Case where valid pairs exist
    if torch.unique(labels).size(0) < labels.size(0):  # There are valid pairs
        loss = compute_contrastive_loss(features, labels)
        assert loss.item() > 0.0, "Loss should be non-zero when valid pairs exist"
