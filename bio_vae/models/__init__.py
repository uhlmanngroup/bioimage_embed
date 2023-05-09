import torch
import torch.nn.functional as F
# Note - you must have torchvision installed for this example
from torch.utils.data import DataLoader

from bio_vae.transforms import DistogramToMaskPipeline

from .ae import AutoEncoder
# from .vae_bio import Mask_VAE, Image_VAE
from .utils import BaseVAE
from .vae import VAE
from .vq_vae import VQ_VAE

from .bolts import ResNet18VAEEncoder,ResNet18VAEDecoder

class Bio_VAE(BaseVAE):
    model_lookup = {
        "vq_vae": VQ_VAE,
        "vae": VAE,
    }
    # model_defaults = {VQ_VAE:{"channels":1},
    #                   VAE: {}}
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    
    def init_model(self,model,*args,**kwargs):
        if type(model) is str:
            return self.model_lookup[model.lower()](*args, **kwargs)
        else:
            return model
        
    # def init_pythae_model(self,model,*args,**kwargs):
        # return model
              
    def __init__(self, model="VQ_VAE", backend="", *args, **kwargs):
        super(Bio_VAE, self).__init__()
        # if backend=="":
        self.model = self.init_model(model,*args,**kwargs)
        self.model_name = self.model._get_name()
        # if backend=="pythae":
            # self.model = self.init_pythae_model(model,*args,**kwargs)

    # def __getattr__(self, attr):
    #     return getattr(self.obj, attr)
    
    def update(self):
        pass
    
    def img_to_latent(self,img):   
        vq = self.get_model().model._vq_vae
        embedding_torch = vq._embedding
        embedding_tensor_in = self.get_model().model.encoder_z(img)
        embedding_tensor_out = vq(embedding_tensor_in)
        latent = embedding_torch(embedding_tensor_out[-1].argmax(axis=1))
        return latent
    
    def forward(self, x,epoch=None):
        return self.model(x,epoch=None)

    def decoder(self, z):
        return self.model.decoder(z)

    def encoder(self, img):
        return self.model.encoder(img)

    def decode(self, z):
        return self.model.decode(z)

    def encode(self, img):
        return self.model.encode(img)

    def recon(self, img):
        return self.model.recon({"data":img})

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
    
    def image_from_latent(self, z, window_size):
        # This should be class-method based
        # I.e. self.decoder(z)
        image = self.decoder(z).detach().numpy()
        return image

# class Image_VAE(BaseVAE):
#     model_lookup = {
#         "vq_vae": VQ_VAE,
#         "vae": VAE,
#     }
#     # model_defaults = {VQ_VAE:{"channels":1},
#     #                   VAE: {}}
#     # by default our latent space is 50-dimensional
#     # and we use 400 hidden units
#     def __init__(self, model="VQ_VAE", *args, **kwargs):
#         super(Image_VAE, self).__init__()
#         if type(model) is str:
#             self.model = self.model_lookup[model.lower()](*args, **kwargs)
#         else:
#             self.model = model

#     # def __getattr__(self, attr):
#     #     return getattr(self.obj, attr)

#     def forward(self, x):
#         return self.model(x)

#     def decoder(self, z):
#         return self.model.decoder(z)

#     def encoder(self, img):
#         return self.model.encoder(img)

#     def decode(self, z):
#         return self.model.decode(z)

#     def encode(self, img):
#         return self.model.encode(img)

#     def recon(self, img):
#         return self.model.recon(img)

#     def image_from_latent(self, z, window_size):
#         # This should be class-method based
#         # I.e. self.decoder(z)
#         image = self.decoder(z).detach().numpy()
#         return image

#     def get_embedding(self):
#         return self.model.get_embedding()

#     def loss_function(self, *args, recons, input, distance_matrix_loss=True, **kwargs):

#         # decode_z, input, mu, log_var = kwargs
#         # # Check to see if distance matrix creates a shape without intersecting edges
#         # x_diff = torch.diff(recons,1,-1)-torch.diff(recons,2,-1)
#         # y_diff = torch.diff(recons,1,-2)

#         # Need to invent metric for ensuring that the final shape is a simple polygon

#         diag_loss = F.mse_loss(
#             torch.diagonal(recons), torch.zeros_like(torch.diagonal(recons))
#         )
#         symmetry_loss = F.mse_loss(recons, recons.transpose(3, 2))
#         vae_loss = self.model.loss_function(*args, recons=recons, input=input, **kwargs)
#         if distance_matrix_loss:
#             vae_loss["loss"] = (
#                 1 / 3 * vae_loss["loss"] + 1 / 3 * diag_loss + 1 / 4 * symmetry_loss
#             )

#         return vae_loss
