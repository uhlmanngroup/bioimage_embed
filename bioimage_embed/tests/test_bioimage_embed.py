import pytest
import torch
from ..bie import BioImageEmbed


@pytest.fixture()
def test_bioimage_embed():
    bie = BioImageEmbed()
    bie.train()
    bie.infer()
    bie.validate()
    assert bie(torch.tensor([1, 2, 3, 4, 5]))
    assert bie.model(torch.tensor([1, 2, 3, 4, 5]))

    bie.model(torch.tensor([1, 2, 3, 4, 5]))
