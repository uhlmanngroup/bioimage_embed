import torch
import torch.nn.functional as F
# Note - you must have torchvision installed for this example
from torch.utils.data import DataLoader


from .ae import AutoEncoder
# from .vae_bio import Mask_VAE, Image_VAE
from .utils import BaseVAE
from .resnets import VAE
# from .vae import VAE
from .vq_vae import VQ_VAE

from ..bolts import ResNet18VAEEncoder,ResNet18VAEDecoder
from torch import nn
class Bio_VAE(nn.Module):
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
    
    def img_to_latent_vq(self,img):   
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

    def get_embedding(self):
        return self.model.get_embedding()

    def output_from_results(self, *args, **kwargs):
        return self.model.output_from_results(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.model.sample(*args, **kwargs)
    
    def image_from_latent(self, z, window_size):
        # This should be class-method based
        # I.e. self.decoder(z)
        image = self.decoder(z).detach().numpy()
        return image
