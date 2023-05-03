from torch import nn
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder

from pythae.models import VQVAE, VQVAEConfig, VAE, VAEConfig

from pl_bolts.models.autoencoders import (
    resnet18_encoder,
    resnet18_decoder,
    resnet50_decoder,
    resnet50_encoder,
)

class ResNet18VAEEncoder(BaseEncoder):

    enc_out_dim = 512

    def __init__(self, model_config:VAEConfig, first_conv=False, maxpool1=False,**kwargs):
        super(ResNet18VAEEncoder, self).__init__()
        
        input_height = model_config.input_dim[-2]
        latent_dim = model_config.latent_dim
        
        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.embedding = nn.Linear(self.enc_out_dim, latent_dim)
        self.log_var = nn.Linear(self.enc_out_dim, latent_dim) 
        # self.fc1 = nn.Linear(512, latent_dim)
        # self._adaptive_pool = nn.AdaptiveAvgPool2d((embedding_dim, embedding_dim))

    def forward(self, x):
        output = ModelOutput()
        x = self.encoder(x)
        # x = self.fc1(x)
        output["embedding"] = self.embedding(x)
        output["log_covariance"] = self.log_var(x)
        return output


class ResNet18VAEDecoder(BaseDecoder):
    def __init__(self, model_config:VAEConfig, first_conv=False, maxpool1=False,**kwargs):
        super(ResNet18VAEDecoder, self).__init__()
        latent_dim = model_config.latent_dim
        input_height = model_config.input_dim[-2]
        self.decoder = resnet18_decoder(latent_dim, input_height, first_conv, maxpool1)

    def forward(self, x):
        output = ModelOutput()
        # output = ModelOutput()
        x = self.decoder(x)
        # x = self.fc1(x)
        output["reconstruction"] = x
        return output


# class ResNet18VAE(VAE):
#     def 