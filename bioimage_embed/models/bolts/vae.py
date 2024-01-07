from torch import nn
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder


from pythae import models
from pythae.models import VQVAEConfig, VAEConfig
from pl_bolts.models import autoencoders
from pythae.models import VQVAE, VQVAEConfig, VAE, VAEConfig

from pl_bolts.models.autoencoders import (
    resnet18_encoder,
    resnet18_decoder,
    resnet50_decoder,
    resnet50_encoder,
)


class ResNet50VAEEncoder(BaseEncoder):
    enc_out_dim = 2048

    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet50VAEEncoder, self).__init__()

        input_height = model_config.input_dim[-2]
        latent_dim = model_config.latent_dim

        self.encoder = resnet50_encoder(first_conv, maxpool1)
        self.embedding = nn.Linear(self.enc_out_dim, latent_dim)
        self.log_var = nn.Linear(self.enc_out_dim, latent_dim)
        # self.fc1 = nn.Linear(512, latent_dim)
        # self._adaptive_pool = nn.AdaptiveAvgPool2d((embedding_dim, embedding_dim))

    def forward(self, x):
        output = ModelOutput()
        x = self.encoder(x)
        # x = self.fc1(x)
        return ModelOutput(embedding=self.embedding(x), log_covariance=self.log_var(x))


class ResNet50VAEDecoder(BaseDecoder):
    enc_out_dim = 2048
    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet50VAEDecoder, self).__init__()
        latent_dim = model_config.latent_dim
        input_height = model_config.input_dim[-2]
        self.embedding = nn.Linear(latent_dim, self.enc_out_dim)
        self.decoder = resnet50_decoder(self.enc_out_dim, input_height, first_conv, maxpool1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.decoder(x)
        return ModelOutput(reconstruction=x)


class ResNet18VAEEncoder(BaseEncoder):
    enc_out_dim = 512

    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet18VAEEncoder, self).__init__()

        input_height = model_config.input_dim[-2]
        latent_dim = model_config.latent_dim

        self.encoder = resnet18_encoder(first_conv, maxpool1)
        self.embedding = nn.Linear(self.enc_out_dim, latent_dim)
        self.log_var = nn.Linear(self.enc_out_dim, latent_dim)
        # self.fc1 = nn.Linear(512, latent_dim)
        # self._adaptive_pool = nn.AdaptiveAvgPool2d((embedding_dim, embedding_dim))

    def forward(self, x):
        x = self.encoder(x)
        # x = self.fc1(x)

        return ModelOutput(embedding=self.embedding(x), log_covariance=self.log_var(x))


class ResNet18VAEDecoder(BaseDecoder):
    enc_out_dim = 512

    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet18VAEDecoder, self).__init__()
        latent_dim = model_config.latent_dim
        input_height = model_config.input_dim[-2]
        self.decoder = resnet18_decoder(
            self.enc_out_dim, input_height, first_conv, maxpool1
        )
        self.embedding = nn.Linear(latent_dim, self.enc_out_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.decoder(x)
        return ModelOutput(reconstruction=x)


class VAEPythaeWrapper(models.VAE):
    def __init__(
        self,
        model_config,
        input_height,
        enc_type="resnet18",
        enc_out_dim=512,
        first_conv=False,
        maxpool1=False,
        kl_coeff=0.1,
        encoder=None,
        decoder=None,
    ):
        super(models.BaseAE, self).__init__()
        self.model_name = "VAE_bolt"
        self.model_config = model_config
        self.model = autoencoders.VAE(
            input_height=input_height,
            enc_type=enc_type,
            enc_out_dim=enc_out_dim,
            first_conv=first_conv,
            maxpool1=maxpool1,
            kl_coeff=kl_coeff,
            latent_dim=model_config.latent_dim,
        )
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder

    def forward(self, x, epoch=None):
        # return ModelOutput(x=x,recon_x=x,z=x,loss=1)
        # # Forward pass logic
        x = x["data"]
        x_recon = self.model(x)
        z, recon_x, p, q = self.model._run_step(x)
        loss, logs = self.model.step((x, x), batch_idx=epoch)

        return ModelOutput(recon_x=recon_x, z=z, logs=logs, loss=loss)
