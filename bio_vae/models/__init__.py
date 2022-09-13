from .ae import AutoEncoder
from .vae import VAE
from .vq_vae import VQ_VAE

# from .vae_bio import Mask_VAE, Image_VAE
from .utils import BaseVAE


# Note - you must have torchvision installed for this example
from torch.utils.data import DataLoader
import torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bio_vae.transforms import DistogramToMaskPipeline
from .utils import BaseVAE
from bio_vae.models import VQ_VAE, VAE


class Mask_VAE(BaseVAE):
    model_lookup = {
        "vq_vae": VQ_VAE,
        "vae": VAE,
    }
    # model_defaults = {VQ_VAE:{"channels":1},
    #                   VAE: {}}
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, model="VQ_VAE", *args, **kwargs):
        super(Mask_VAE, self).__init__()
        if type(model) is str:
            self.model = self.model_lookup[model.lower()](*args, **kwargs)
        else:
            self.model = model

    # def __getattr__(self, attr):
    #     return getattr(self.obj, attr)

    def forward(self, x):
        return self.model(x)

    def decoder(self, z):
        return self.model.decoder(z)

    def encoder(self, img):
        return self.model.encoder(img)

    def decode(self, z):
        return self.model.decode(z)

    def encode(self, img):
        return self.model.encode(img)

    def recon(self, img):
        return self.model.recon(img)

    def mask_from_latent(self, z, window_size):
        # This should be class-method based
        # I.e. self.decoder(z)
        dist = self.decoder(z).detach().numpy()
        mask = DistogramToMaskPipeline(window_size)(dist)
        return mask

    def get_embedding(self):
        return self.model.get_embedding()

    def loss_function(self, *args, recons, input, distance_matrix_loss=True, **kwargs):

        # decode_z, input, mu, log_var = kwargs
        # # Check to see if distance matrix creates a shape without intersecting edges
        # x_diff = torch.diff(recons,1,-1)-torch.diff(recons,2,-1)
        # y_diff = torch.diff(recons,1,-2)

        # Need to invent metric for ensuring that the final shape is a simple polygon

        diag_loss = F.mse_loss(
            torch.diagonal(recons), torch.zeros_like(torch.diagonal(recons))
        )
        symmetry_loss = F.mse_loss(recons, recons.transpose(3, 2))
        vae_loss = self.model.loss_function(*args, recons=recons, input=input, **kwargs)
        if distance_matrix_loss:
            vae_loss["loss"] = (
                8 / 10 * vae_loss["loss"] + 1 / 10 * diag_loss + 1 / 10 * symmetry_loss
            )

        return vae_loss

    def output_from_results(self, *args, **kwargs):
        return self.model.output_from_results(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)


class Image_VAE(BaseVAE):
    model_lookup = {
        "vq_vae": VQ_VAE,
        "vae": VAE,
    }
    # model_defaults = {VQ_VAE:{"channels":1},
    #                   VAE: {}}
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, model="VQ_VAE", *args, **kwargs):
        super(Image_VAE, self).__init__()
        if type(model) is str:
            self.model = self.model_lookup[model.lower()](*args, **kwargs)
        else:
            self.model = model

    # def __getattr__(self, attr):
    #     return getattr(self.obj, attr)

    def forward(self, x):
        return self.model(x)

    def decoder(self, z):
        return self.model.decoder(z)

    def encoder(self, img):
        return self.model.encoder(img)

    def decode(self, z):
        return self.model.decode(z)

    def encode(self, img):
        return self.model.encode(img)

    def recon(self, img):
        return self.model.recon(img)

    def image_from_latent(self, z, window_size):
        # This should be class-method based
        # I.e. self.decoder(z)
        image = self.decoder(z).detach().numpy()
        return image

    def get_embedding(self):
        return self.model.get_embedding()

    def loss_function(self, *args, recons, input, distance_matrix_loss=True, **kwargs):

        # decode_z, input, mu, log_var = kwargs
        # # Check to see if distance matrix creates a shape without intersecting edges
        # x_diff = torch.diff(recons,1,-1)-torch.diff(recons,2,-1)
        # y_diff = torch.diff(recons,1,-2)

        # Need to invent metric for ensuring that the final shape is a simple polygon

        diag_loss = F.mse_loss(
            torch.diagonal(recons), torch.zeros_like(torch.diagonal(recons))
        )
        symmetry_loss = F.mse_loss(recons, recons.transpose(3, 2))
        vae_loss = self.model.loss_function(*args, recons=recons, input=input, **kwargs)
        if distance_matrix_loss:
            vae_loss["loss"] = (
                1 / 3 * vae_loss["loss"] + 1 / 3 * diag_loss + 1 / 4 * symmetry_loss
            )

        return vae_loss
