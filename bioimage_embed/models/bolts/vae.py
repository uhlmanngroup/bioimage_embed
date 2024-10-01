from torch import nn
from transformers.utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder


from pythae.models import VAEConfig
from . import resnets


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResNet50VAEEncoder(BaseEncoder):
    enc_out_dim = 2048

    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet50VAEEncoder, self).__init__()

        # input_height = model_config.input_dim[-2]
        latent_dim = model_config.latent_dim

        self.encoder = resnets.resnet50_encoder(first_conv, maxpool1)
        self.embedding = nn.Linear(self.enc_out_dim, latent_dim)
        self.log_var = nn.Linear(self.enc_out_dim, latent_dim)
        # self.fc1 = nn.Linear(512, latent_dim)
        # self._adaptive_pool = nn.AdaptiveAvgPool2d((embedding_dim, embedding_dim))

    def forward(self, x):
        x = self.encoder(x)
        # x = self.fc1(x)
        return ModelOutput(embedding=self.embedding(x), log_covariance=self.log_var(x))


class ResNet50VAEDecoder(BaseDecoder):
    enc_out_dim = 512

    def __init__(
        self, model_config: VAEConfig, first_conv=False, maxpool1=False, **kwargs
    ):
        super(ResNet50VAEDecoder, self).__init__()
        latent_dim = model_config.latent_dim
        input_height = model_config.input_dim[-2]
        self.embedding = nn.Linear(latent_dim, self.enc_out_dim)
        self.decoder = resnets.resnet50_decoder(
            self.enc_out_dim, input_height, first_conv, maxpool1
        )

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

        # input_height = model_config.input_dim[-2]
        latent_dim = model_config.latent_dim

        self.encoder = resnets.resnet18_encoder(first_conv, maxpool1)
        self.embedding = nn.Linear(self.enc_out_dim, latent_dim)
        self.log_var = nn.Linear(self.enc_out_dim, latent_dim)

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
        self.decoder = resnets.resnet18_decoder(
            self.enc_out_dim, input_height, first_conv, maxpool1
        )
        self.embedding = nn.Linear(latent_dim, self.enc_out_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.decoder(x)
        return ModelOutput(reconstruction=x)
