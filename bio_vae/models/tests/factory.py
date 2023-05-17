from bio_vae.models import create_model
import pytest
import torch

from bio_vae.models import MODELS

# MODELS = ["resnet18_vae", "resnet50_vae", "resnet18_vqvae", "resnet50_vqvae"]
input_dim = (3,256,256)
latent_dim = 50
pretrained = False
progress = True

@pytest.mark.parametrize("model", MODELS)
def test_create_model(model):
    model = create_model(model, input_dim, latent_dim, pretrained, progress)
    data = torch.rand(1, *input_dim)
    output = generated_model(data)
    assert output.size() == data.size()
