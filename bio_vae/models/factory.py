# import torch
# import torch.nn.functional as F

# Note - you must have torchvision installed for this example
# from torch.utils.data import DataLoader

# from bio_vae.transforms import DistogramToMaskPipeline


# from .vae_bio import Mask_VAE, Image_VAE

# from .bolts import ResNet18VAEEncoder, ResNet18VAEDecoder

import pythae

from . import bolts
class ModelFactory:
    def __init__(
        self, input_dim, latent_dim, pretrained=False, progress=True, **kwargs
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.pretrained = pretrained
        self.progress = progress
        self.kwargs = kwargs

    def create_model(self, model_class, encoder_class, decoder_class):
        model_config = model_class(input_dim=self.input_dim, latent_dim=self.latent_dim)
        encoder = encoder_class(model_config)
        decoder = decoder_class(model_config)
        # TODO Fix this
        model = pythae.models.VAE(model_config, encoder=encoder, decoder=decoder)
        return model

    def resnet18_vae(self):
        return self.create_model(
            pythae.models.VAEConfig, bolts.ResNet18VAEEncoder, bolts.ResNet18VAEDecoder
        )

    def resnet50_vae(self):
        return self.create_model(
            pythae.models.VAEConfig, bolts.ResNet50VAEEncoder, bolts.ResNet50VAEDecoder
        )

    def resnet18_vqvae(self):
        return self.create_model(
            pythae.models.VQVAEConfig,
            bolts.ResNet18VQVAEEncoder,
            bolts.ResNet18VQVAEDecoder,
        )

    def resnet50_vqvae(self):
        return self.create_model(
            pythae.models.VQVAEConfig,
            bolts.ResNet50VQVAEEncoder,
            bolts.ResNet50VQVAEDecoder,
        )


MODELS = ["resnet18_vae", "resnet50_vae", "resnet18_vqvae", "resnet50_vqvae"]


def create_model(model, input_dim, latent_dim, pretrained=False, progress=True):
    factory = ModelFactory(input_dim, latent_dim, pretrained, progress)
    return getattr(factory, model)()