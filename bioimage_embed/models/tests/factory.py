from bio_vae.models import create_model
import pytest
import torch

from bio_vae.models import MODELS

# MODELS = ["resnet18_vae", "resnet50_vae", "resnet18_vqvae", "resnet50_vqvae"]

input_dim = [(3,256,256),(3,224,224),(1,256,256),(1,224,224)]
latent_dim = [50,16]
pretrained = False
progress = True

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("latent_dim", latent_dim)
@pytest.mark.parametrize("input_dim", input_dim)
def test_create_model(model, input_dim, latent_dim, pretrained, progress):
    model = create_model(model, input_dim, latent_dim, pretrained, progress)
    data = torch.rand(1, *input_dim)
    output = generated_model(data)
    assert output.size() == data.size()
