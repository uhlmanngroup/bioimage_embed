from mask_vae import datasets
import pytest


@pytest.mark.parametrize(
    "dataset",
    [
        datasets.BBBC010,
    ],
)
def test_download(dataset):
    data = dataset()
    assert len(data) > 0
    assert data[0] is not None


@pytest.mark.parametrize(
    "dataset",
    [
        "BBBC010",
    ],
)
def test_download(dataset):
    data = datasets.BroadDataset(dataset)
    assert len(data) > 0
    assert data[0] is not None
