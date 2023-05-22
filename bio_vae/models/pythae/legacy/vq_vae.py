# TODO make this a relative import
from bio_vae.models import legacy

from torch import nn
from pythae import models

import bio_vae

# from pythae.models import VQVAE, Encoder, Decoder
from pythae.models.base.base_utils import ModelOutput

# import VQVAE, Encoder, Decoder
from pythae.models.nn import BaseDecoder, BaseEncoder
from ...nets.resnet import ResnetDecoder, ResnetEncoder


class Encoder(BaseEncoder):
    def __init__(self, model_config, **kwargs):

        embedding_dim = model_config.latent_dim
        input_dim = model_config.input_dim[1:]

        self.model = ResnetEncoder(
            num_hiddens=model_config.num_hiddens,
            in_channels=model_config.input_dim[0],
            num_residual_layers=model_config.num_residual_layers,
            num_residual_hiddens=model_config.num_residual_hiddens,
        )

    def forward(self, x):
        # TODO check for typo in pre_quantized
        return ModelOutput(pre_qantized=self.model(x["data"]))


class Decoder(BaseDecoder):
    def __init__(self, model_config, **kwargs):
        self.model = ResnetDecoder(
            in_channels=model_config.latent_dim,
            num_hiddens=model_config.num_hiddens,
            num_residual_layers=model_config.num_residual_layers,
            num_residual_hiddens=model_config.num_residual_hiddens,
            out_channels=model_config.input_dim[0],
        )

    def forward(self, x):
        reconstruction = self.model(x["embedding"])
        return ModelOutput(reconstruction=reconstruction)


class VQVAE(models.VQVAE):
    def __init__(self, model_config, **kwargs):
        super(models.BaseAE, self).__init__()
        # super(nn.Module)
        # input_dim (tuple) – The input_data dimension.

        self.model_name = "VQVAE"
        self.model_config = model_config
        self.model = legacy.vq_vae.VQ_VAE(
            channels=model_config.input_dim[0],
            embedding_dim=model_config.latent_dim,
            **{**vars(model_config), **kwargs}
        )
        self.encoder = self.model._encoder
        self.decoder = self.model._decoder
        # This isn't completely necessary for training I don't think
        # self._set_quantizer(model_config)
        self.quantizer = self.model._vq_vae

    def forward(self, x, epoch=None):
        # loss, x_recon, perplexity = self.model.forward(x["data"])
        z = self.model.encoder(x["data"])
        z = self.model._pre_vq_conv(z)
        loss, quantized, perplexity, encodings = self.model._vq_vae(z)
        x_recon = self.model._decoder(quantized)
        # return loss, x_recon, perplexity
        loss_dict = self.model.loss_function(
            loss,
            x_recon,
            perplexity,
            vq_loss=loss,
            perplexity=perplexity,
            recons=x_recon,
            input=x["data"],
        )
        indices = (encodings == 1).nonzero(as_tuple=True)
        # self.model.encode()

        return ModelOutput(
            # recon_loss=recon_loss,
            vq_loss=loss,
            # loss=loss,
            recon_x=x_recon,
            z=quantized,
            quantized_indices=indices[0],
            **loss_dict
        )
        # return ModelOutput(reconstruction=x_recon, **loss_dict)
