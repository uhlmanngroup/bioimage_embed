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
    return F.mse_loss(distance_matrix, distance_matrix.transpose(-2, -1))

# def symmetry_loss(distance_matrix):
#     return F.mse_loss(distance_matrix, torch.transpose(distance_matrix, dim0=-2, dim1=-1))



class MaskEmbed(LitAutoEncoderTorch):
    def __init__(self, model,args=None):
        super().__init__(model,args)
    def batch_to_tensor(self, batch):
        return super().batch_to_tensor(batch[0].float())
    
    def loss_function(self, model_output, *args, **kwargs):
        loss = model_output.loss
        loss = loss + diagonal_loss(model_output.recon_x)
        loss = loss + symmetry_loss(model_output.recon_x)
        return loss
    

    # def embedding_from_output(self,model_output):
    #     return model_output.z.view(model_output.z.shape[0], -1)