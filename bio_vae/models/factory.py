# import torch
# import torch.nn.functional as F

# Note - you must have torchvision installed for this example
# from torch.utils.data import DataLoader

# from bio_vae.transforms import DistogramToMaskPipeline


# from .vae_bio import Mask_VAE, Image_VAE

# from .bolts import ResNet18VAEEncoder, ResNet18VAEDecoder

import pythae
from .pythae import legacy
from . import bolts
from functools import partial


class ModelFactory:
    def __init__(
        self, input_dim, latent_dim, pretrained=False, progress=True, **kwargs
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.pretrained = pretrained
        self.progress = progress
        self.kwargs = kwargs

    def create_model(
        self, model_config_class, model_class, encoder_class, decoder_class
    ):
        model_config = model_config_class(
            input_dim=self.input_dim, latent_dim=self.latent_dim, **self.kwargs
        )
        encoder = encoder_class(model_config)
        decoder = decoder_class(model_config)
        # TODO Fix this
        return model_class(model_config=model_config, encoder=encoder, decoder=decoder)

    def resnet18_vae(self):
        return self.create_model(
            pythae.models.VAEConfig,
            pythae.models.VAE,
            bolts.ResNet18VAEEncoder,
            bolts.ResNet18VAEDecoder,
        )

    def resnet50_vae(self):
        return self.create_model(
            pythae.models.VAEConfig,
            pythae.models.VAE,
            bolts.ResNet50VAEEncoder,
            bolts.ResNet50VAEDecoder,
        )

    def resnet18_vqvae(self):
        return self.create_model(
            pythae.models.VQVAEConfig,
            pythae.models.VQVAE,
            bolts.ResNet18VQVAEEncoder,
            bolts.ResNet18VQVAEDecoder,
        )

    def resnet50_vqvae(self):
        return self.create_model(
            pythae.models.VQVAEConfig,
            pythae.models.VQVAE,
            bolts.ResNet50VQVAEEncoder,
            bolts.ResNet50VQVAEDecoder,
        )

    def resnet_vqvae_legacy(self, depth):
        return self.create_model(
            partial(pythae.models.VQVAEConfig, **self.kwargs),
            # partial(legacy.vq_vae.VQVAE,**self.kwargs,num_hidden_residuals=depth),
            partial(legacy.vq_vae.VQVAE, num_hidden_residuals=depth, **self.kwargs),
            encoder_class=lambda x: None,
            decoder_class=lambda x: None,
        )

    def resnet18_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(18)

    def resnet50_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(50)

    def resnet101_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(101)

    def resnet150_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(150)

    def resnet152_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(152)


MODELS = [
    "resnet18_vae",
    "resnet50_vae",
    "resnet18_vqvae",
    "resnet50_vqvae",
    "resnet18_vqvae_legacy",
    "resnet50_vqvae_legacy",
    "resnet11_vqvae_legacy",
    "resnet150_vqvae_legacy",
    "resnet152_vqvae_legacy",
]


def create_model(
    model, input_dim, latent_dim, pretrained=False, progress=True, **kwargs
):
    factory = ModelFactory(input_dim, latent_dim, pretrained, progress, **kwargs)
    return getattr(factory, model)()
