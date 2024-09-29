# import torch

# import torch.nn.functional as F

# Note - you must have torchvision installed for this example
# from torch.utils.data import DataLoader

# from bioimage_embed.transforms import DistogramToMaskPipeline


# from .vae_bio import Mask_VAE, Image_VAE

# from .bolts import ResNet18VAEEncoder, ResNet18VAEDecoder

from typing import Tuple
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
            input_dim=tuple(self.input_dim),
            latent_dim=self.latent_dim,
        )
        encoder = encoder_class(model_config)
        decoder = decoder_class(model_config)
        # TODO Fix this
        return model_class(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
        )

    def dummy_model(self):
        return self.create_model(
            pythae.models.VAEConfig,
            pythae.models.VAE,
            lambda x: None,
            lambda x: None,
        )

    def resnet_vae_bolt(
        self,
        enc_type,
        enc_out_dim,
        first_conv=False,
        maxpool1=False,
        kl_coeff=1.0,
    ):
        return self.create_model(
            pythae.models.VAEConfig,
            partial(
                bolts.vae.VAEPythaeWrapper,
                input_height=self.input_dim[1],
                enc_type=enc_type,
                enc_out_dim=enc_out_dim,
                first_conv=first_conv,
                maxpool1=maxpool1,
                kl_coeff=kl_coeff,
            ),
            encoder_class=lambda x: None,
            decoder_class=lambda x: None,
        )

    # bolts.vae.VAEPythaeWrapper(
    #         input_height=self.input_dim[1],
    #         enc_type=enc_type,
    #         enc_out_dim=enc_out_dim,
    #         first_conv=first_conv,
    #         maxpool1=maxpool1,
    #         kl_coeff=kl_coeff,
    #         latent_dim=self.latent_dim,
    #     )

    def resnet18_vae_bolt(self, **kwargs):
        return self.resnet_vae_bolt(enc_type="resnet18", enc_out_dim=512, **kwargs)

    def resnet50_vae_bolt(self, **kwargs):
        return self.resnet_vae_bolt(enc_type="resnet50", enc_out_dim=2048, **kwargs)

    def resnet18_vae(self):
        return self.create_model(
            partial(
                pythae.models.VAEConfig,
                use_default_encoder=False,
                use_default_decoder=False,
                **self.kwargs,
            ),
            pythae.models.VAE,
            bolts.ResNet18VAEEncoder,
            bolts.ResNet18VAEDecoder,
        )

    def resnet18_vqvae(self):
        return self.create_model(
            partial(
                pythae.models.VQVAEConfig,
                use_default_encoder=False,
                use_default_decoder=False,
                **self.kwargs,
            ),
            pythae.models.VQVAE,
            bolts.ResNet18VQVAEEncoder,
            bolts.ResNet18VQVAEDecoder,
        )

    def resnet18_beta_vae(self):
        return self.create_model(
            partial(
                pythae.models.BetaVAEConfig,
                use_default_encoder=False,
                use_default_decoder=False,
                **self.kwargs
            ),
            pythae.models.BetaVAE,
            bolts.ResNet18VAEEncoder,
            bolts.ResNet18VAEDecoder,
        )

    def resnet50_vae(self):
        return self.create_model(
            partial(
                pythae.models.VAEConfig,
                use_default_encoder=False,
                use_default_decoder=False,
                **self.kwargs
            ),
            pythae.models.VAE,
            bolts.ResNet50VAEEncoder,
            bolts.ResNet50VAEDecoder,
        )

    def resnet50_vqvae(self):
        return self.create_model(
            partial(
                pythae.models.VQVAEConfig,
                use_default_encoder=False,
                use_default_decoder=False,
                **self.kwargs,
            ),
            pythae.models.VQVAE,
            bolts.ResNet50VQVAEEncoder,
            bolts.ResNet50VQVAEDecoder,
        )

    def resnet50_beta_vae(self):
        return self.create_model(
            partial(
                pythae.models.BetaVAEConfig,
                use_default_encoder=False,
                use_default_decoder=False,
                **self.kwargs
            ),
            pythae.models.BetaVAE,
            bolts.ResNet50VAEEncoder,
            bolts.ResNet50VAEDecoder,
        )

    def resnet_vae_legacy(self, depth):
        return self.create_model(
            pythae.models.VAEConfig,
            partial(legacy.VAE, num_residual_hiddens=depth),
            encoder_class=lambda x: None,
            decoder_class=lambda x: None,
        )

    def resnet18_vae_legacy(self):
        return self.resnet_vae_legacy(18)

    def resnet50_vae_legacy(self):
        return self.resnet_vae_legacy(50)

    def resnet_vqvae_legacy(self, depth):
        return self.create_model(
            pythae.models.VQVAEConfig,
            # partial(legacy.vq_vae.VQVAE,**self.kwargs,num_hidden_residuals=depth),
            partial(legacy.vq_vae.VQVAE, depth=depth),
            encoder_class=lambda x: None,
            decoder_class=lambda x: None,
        )

    def resnet18_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(18)

    def resnet50_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(50)

    def resnet101_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(101)

    def resnet110_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(150)

    def resnet152_vqvae_legacy(self):
        return self.resnet_vqvae_legacy(152)

    def o2vae(self):
        from .o2vae.models.decoders.cnn_decoder import CnnDecoder
        from .o2vae.models.encoders_o2.e2scnn import E2SFCNN
        from .o2vae.models.vae import VAE as O2VAE

        # encoder
        q_net = E2SFCNN(
                  n_channels = 1,
                  n_classes = 64 * 2, # bc vae saves mean and stdDev vecors
                  # `name`: 'o2_cnn' for o2-invariant encoder. 'cnn_encoder' for standard cnn encoder.
                  name="o2_cnn_encoder",
                  # `cnn_dims`: must be 6 elements long. Increase numbers for larger model capacity
                  cnn_dims=[6, 9, 12, 12, 19, 25],
                  # `layer_type`: type of cnn layer (following e2cnn library examples)
                  layer_type="inducedgated_norm",  # recommend not changing
                  # `N`: Ignored if `name!='o2'`. Negative means the model will be O2-invariant.
                  #     Again, see (e2cnn library examples). Recommend not changing.
                  N=-3,
                )

        # decoder
        p_net = CnnDecoder(
                  zdim = 64,
                  name="cnn_decoder",  # 'cnn' is the ony option
                  # `cnn_dims`: each extra layer doubles the dimension (image width) by a factor of 2.
                  #    E.g. if there are 6 elements, image width is 2^6=64
                  cnn_dims=[192, 96, 96, 48, 48, 48],
                  #cnn_dims=[192, 96, 96, 48, 48, 24, 24, 12, 12],
                  out_channels=1,
                )

        # vae
        model = O2VAE(
                  q_net = q_net,
                  p_net = p_net,
                  zdim = 64,         # vae bottleneck layer
                  do_sigmoid = True, # whether to make the output be between [0,1]. Usually True. 
                  loss_kwargs = dict(
                    # 'beta' from beta-vae, or the weight on the KL-divergence term https://openreview.net/forum?id=Sy2fzU9gl
                    beta=0.01,
                    # `recon_loss_type`: "bce" (binary cross entropy) or "mse" (mean square error)
                    #    or "ce" (cross-entropy, but warning, not been tested well)
                    #recon_loss_type="bce",
                    recon_loss_type="mse",
                    # for reconstrutcion loss, pixel mask. Must be either `None` or an array with same dimension as the images.
                    mask=None,
                    align_loss=True,  # whether to align the output image to the input image
                    # whether to use efficient Foureier-based loss alignment. (Ignored if align_loss==False)
                    align_fourier=True,
                    # whether to do align the best rotation AND flip, instead of just rotation. (Ignored if align_loss==False)
                    do_flip=True,
                    # if doing brute force align loss, this is the rotation discretization. (Ignored if
                    #   align_loss==False or if align_fourier==True)
                    rot_steps=2,
                    # Recommend not changing. The vae prior distribution. Optoins: ("standard","normal","gmm"). See models.vae.VAE for deatils.
                    prior_kwargs=dict( prior="standard",),
                  )
                )

        # extra attributes
        model.encoder = q_net
        model.decoder = p_net

        return model


    #    return getattr(self
    #         (
    #             self.input_dim, self.latent_dim, self.pretrained, self.progress),
    #         ),
    #         model,
    #     )


__all_models__ = [
    "resnet18_vae",
    "resnet18_beta_vae",
    "resnet50_vae",
    "resnet50_beta_vae",
    "resnet18_vae_bolt",
    "resnet50_vae_bolt",
    "resnet18_vqvae",
    "resnet50_vqvae",
    "resnet18_vqvae_legacy",
    "resnet50_vqvae_legacy",
    "resnet101_vqvae_legacy",
    "resnet110_vqvae_legacy",
    "resnet152_vqvae_legacy",
    "resnet18_vae_legacy",
    "resnet50_vae_legacy",
    "o2vae",
]

__all_small_models__ = [
    "resnet18_vae",
    "resnet18_vqvae",
    "dummy_model",
]


def create_model(
    model: str,
    input_dim: Tuple[int, int, int],
    latent_dim: int,
    pretrained=False,
    progress=True,
    **kwargs,
):
    factory = ModelFactory(input_dim, latent_dim, pretrained, progress, **kwargs)
    return getattr(factory, model)()
