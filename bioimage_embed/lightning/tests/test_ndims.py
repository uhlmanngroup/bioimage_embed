import pytest
import torch

# Fixture for batch sizes
@pytest.fixture(params=[1, 16])
def batch_size(request):
    return request.param


# Fixture for channel sizes
@pytest.fixture(params=[1, 3, 5])
def channel_size(request):
    return request.param


# Fixture for extra dimen
@pytest.fixture(params=[2**i for i in range(2, 9)])
def extra_dim(request):
    return request.param


# Fixture for generating tensors
@pytest.fixture
def input_tensor(batch_size, channel_size, extra_dim):
    return torch.randn(batch_size, channel_size, extra_dim)


# Test for channel aware losses
@pytest.mark.skip(reason="Not implemented")
def test_ndim_loss(input_tensor, loss_fn):
    # Call the function and make assertions
    loss = loss_fn(input_tensor)
    assert loss is not None  # Example assertion
    assert loss is not None
    assert loss is not torch.nan
    assert loss is not torch.inf
    assert loss is not -torch.inf
    assert loss.min() >= 0
    assert len(loss.shape) == 0
