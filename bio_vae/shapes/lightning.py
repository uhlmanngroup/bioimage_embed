import torch
import torchvision
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

from ..lightning import LitAutoEncoderTorch

def diagonal_loss(distance_matrix):
    return F.mse_loss(
        torch.diagonal(distance_matrix),
        torch.zeros_like(torch.diagonal(distance_matrix)),
    )

def symmetry_loss(distance_matrix):
    return F.mse_loss(distance_matrix, distance_matrix.transpose(3, 2))


class Mask_Embed(LitAutoEncoderTorch):
    def __init__(self, model,args=None):
        super(LitAutoEncoderTorch).__init__()

    def loss_function(self, model_output, *args, **kwargs):
        loss = model_output.loss
        loss = loss + diagonal_loss(model_output.recon_x)
        loss = loss + symmetry_loss(model_output.recon_x)
        return loss
