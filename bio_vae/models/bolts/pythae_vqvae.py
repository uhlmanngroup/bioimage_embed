from torch import nn
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
import torch
from pythae.models import VQVAE, VQVAEConfig, VAE, VAEConfig

from pl_bolts.models.autoencoders import (
    resnet18_encoder,
    resnet18_decoder,
    resnet50_decoder,
    resnet50_encoder,
)


class ResNet50VQVAEEncoder(BaseEncoder):

    enc_out_dim = 2048

    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet50VQVAEEncoder, self).__init__()

        input_height = model_config.input_dim[-2]
        latent_dim = model_config.latent_dim

        self.encoder = resnet50_encoder(first_conv, maxpool1)
        self.encoder.avgpool = nn.Identity()
        self.embedding = nn.Linear(self.enc_out_dim, latent_dim)
        self.log_var = nn.Linear(self.enc_out_dim, latent_dim)
        self.prequantized = nn.Conv2d(self.enc_out_dim, latent_dim, 1, 1)
        
        self.fc = nn.Linear(401408, 256)

    def forward(self, x):
        output = ModelOutput()
        # y = self.encoder(x).view(-1,self.enc_out_dim,14,14)
        x = self.encoder(x)
        x = self.fc(x)
        
        x = x.view(x.size(0), -1, 1, 1) 
        embedding = self.prequantized(x.view(-1, self.enc_out_dim, 1, 1))
        # y = self.encoder(x).view(-1,2048,14,14)
        # output["embedding"] = self.prequantized(y)
        # output["embedding"] = self.embedding(x)

        # output["log_covariance"] = self.log_var(x)
        return ModelOutput(embedding=embedding)


class ResNet50VQVAEDecoder(BaseDecoder):
    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet50VQVAEDecoder, self).__init__()
        latent_dim = model_config.latent_dim
        input_height = model_config.input_dim[-1]
        self.decoder = resnet18_decoder(latent_dim, input_height, first_conv, maxpool1)
        # self.decoder.linear = nn.Identity()
        self.postquantized = nn.Conv2d(latent_dim, 512, 1, 1)
        # self.fc = nn.Linear(self.enc_out_dim, latent_dim)

    def forward(self, x):
        output = ModelOutput()
        # a = nn.Conv2d(16,512,1,1)
        # self.postquantized(x)
        x = self.decoder(x.squeeze(-1).squeeze(-1))
        
        return ModelOutput(reconstruction=x)


class ResNet18VQVAEEncoder(BaseEncoder):

    enc_out_dim = 512

    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet18VQVAEEncoder, self).__init__()

        input_height = model_config.input_dim[-2]
        latent_dim = model_config.latent_dim

        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.embedding = nn.Linear(self.enc_out_dim, latent_dim)
        self.log_var = nn.Linear(self.enc_out_dim, latent_dim)
        self.prequantized = nn.Conv2d(self.enc_out_dim, latent_dim, 1, 1)

    def forward(self, x):
        # output = ModelOutput()
        x = self.encoder(x)
        # x = self.fc1(x)
        embedding = self.prequantized(x.view(-1, self.enc_out_dim, 1, 1))
        log_covariance = self.log_var(x)
        return ModelOutput(embedding=embedding, log_covariance=log_covariance)


class ResNet18VQVAEDecoder(BaseDecoder):
    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet18VQVAEDecoder, self).__init__()
        latent_dim = model_config.latent_dim
        input_height = model_config.input_dim[-2]
        self.decoder = resnet18_decoder(latent_dim, input_height, first_conv, maxpool1)

    def forward(self, x):
        x = self.decoder(x)
        # x = self.fc1(x)
        return ModelOutput(reconstruction=x)
