import pytest
import os
from torchvision import transforms
import torch

from mask_vae.datasets import DSB2018
from mask_vae.transforms import ImagetoDistogram, cropCentroid, DistogramtoImage
from mask_vae.models import AutoEncoder, VAE, VQ_VAE
window_size = 96

transformer_crop = transforms.Compose(
    [
        # transforms.ToPILImage(),
        cropCentroid(window_size),
        transforms.ToTensor(),
        # transforms.Normalize(0, 1),
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
        # transforms.RandomCrop((512, 512)),
        # transforms.ConvertImageDtype(torch.bool)

    ]
)

transformer_dist = transforms.Compose(
    [
        transformer_crop,
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        ImagetoDistogram(window_size),
        # transforms.ToPILImage(),
        # transforms.RandomCrop((512, 512)),
        transforms.ConvertImageDtype(torch.float32)
    ]
)


train_dataset_glob = os.path.join(os.path.expanduser("~"),
                                  "data-science-bowl-2018/stage1_train/*/masks/*.png")
# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")
train_dataset = DSB2018(train_dataset_glob, transform=transformer_dist)

def test_models():
    # vae = AutoEncoder(1, 1)
    vae = VQ_VAE(channels=1)
    img = train_dataset[0].unsqueeze(0)
    loss, x_recon, perplexity = vae(img)
    z = vae.encoder(img)
    y_prime = vae.decoder(z)
    # print(f"img_dims:{img.shape} y:_dims:{x_recon.shape}")
    print(f"img_dims:{img.shape} x_recon:_dims:{x_recon.shape} z:_dims:{z.shape}")
