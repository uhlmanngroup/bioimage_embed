from torch.utils.data import DataLoader
import torch
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
import pythae

# https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=fknqLRCvdJ4I


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


# num_hiddens = 64
# num_residual_hiddens = 32
# num_residual_layers = 2

# embedding_dim = 64
# num_embeddings = 512

# commitment_cost = 0.25

# decay = 0.99

# num_hiddens=64
# num_residual_hiddens=32
# num_residual_layers=2
# embedding_dim=64
# num_embeddings=512
# commitment_cost=0.25
# decay=0.99
# channels=1

class Encoder(BaseEncoder):
    def __init__(
        self,
        model_config=None,
        num_hiddens=64,
        num_residual_hiddens=32,
        num_residual_layers=2,
        in_channels=1,
        embedding_dim=64,
    ):
        super(Encoder, self).__init__()
        if model_config is not None:
            embedding_dim = model_config.latent_dim
            num_hiddens = model_config.num_embeddings
            in_channels = model_config.input_dim[0]
            image_dims = model_config.input_dim[1:]
            
        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )

    def forward(self, inputs):
        output = ModelOutput()

        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        x = self._residual_stack(x)
        x = self._pre_vq_conv(x)
        output["embedding"] = x
        return output




# import pythae
class Decoder(BaseDecoder):
    def __init__(
        self,
        model_config=None,
        in_channels=1,
        num_hiddens=64,
        num_residual_hiddens=32,
        num_residual_layers=2,
        out_channels=1,
    ):
        super(Decoder, self).__init__()

        if model_config is not None:
            num_hiddens = model_config.num_embeddings
            in_channels = model_config.latent_dim
            out_channels = model_config.input_dim[0]
            image_dims = model_config.input_dim[1:]
            
        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs):
        output = ModelOutput()

        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)
        output["reconstruction"] = self._conv_trans_2(x)
        return output


class VQ_VAE(nn.Module):
    def __init__(
        self,
        num_hiddens=64,
        num_residual_hiddens=32,
        num_residual_layers=2,
        embedding_dim=64,
        num_embeddings=512,
        commitment_cost=0.25,
        decay=0.99,
        channels=1,
        **kwargs
    ):
        super(VQ_VAE, self).__init__()

        self._encoder = Encoder(
            None,
            num_hiddens, num_residual_layers, num_residual_hiddens, in_channels=channels,embedding_dim=embedding_dim
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
        self._decoder = Decoder(
            None,
            embedding_dim,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            out_channels=channels,
        )

    def forward(self, x, epoch=0):
        z = self.encoder(x["data"])
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)["reconstruction"]
        return loss, x_recon, perplexity

    def recon(self, x):
        loss, x_recon, perplexity = self.forward(x)
        return x_recon

    def encoder_z(self, x):
        # z = self._encoder(x)
        # z = self._pre_vq_conv(z)
        # z = self._encoder(x)["embedding"]
        # z = self._pre_vq_conv(z)
        return self._encoder(x)["embedding"]

    def encoder_zq(self, x):
        # z = self._encoder(x)
        # z = self._pre_vq_conv(z)
        z = self.encoder_z(x)
        loss, z_q, perplexity, _ = self._vq_vae(z)
        return z_q

    def decoder_z(self, z):
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        return x_recon

    def decoder_zq(self, quantized):
        return self._decoder(quantized)

    def encoder(self, x):
        return self.encoder_z(x)
        # return self.encoder_zq(x)

    def decoder(self, z):
        return self.decoder_z(z)
        # return self.decoder_zq(z)

    # def forward(self, x):
    #     vq_output_eval = self._pre_vq_conv(self._encoder(x))
    #     _, quantize, _, _ = self._vq_vae(vq_output_eval)
    #     reconstructions = self._decoder(quantize)

    def model(self, x):
        loss, quantized, perplexity, _ = self.forward(x)
        x_recon = self.decoder(quantized)
        return x_recon

    def get_embedding(self):
        return self._vq_vae._embedding.weight.data.cpu()

    def encode(self, x):
        return [self.encoder(x), None]

    def decode(self, x):
        return self.decoder(x)

    def output_from_results(self, loss, x_recon, perplexity):
        return x_recon

    def loss_function(self, *args, recons, input, **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        vq_loss, recons, perplexity = args
        # recons = args[0]
        # input = args[1]
        # vq_loss = args[2]
        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss, "VQ_Loss": vq_loss}

        # vq_loss, output, perplexity = self.forward(inputs)
        # output = x_recon
        # loss = self.loss_fn(output, inputs)

        # vq_loss, data_recon, perplexity = model(inputs)
        # recon_error = F.mse_loss(output, inputs)
        # recon_error = self.loss_fn(output, inputs)