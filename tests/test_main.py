import pytest
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from torch.utils.data import DataLoader
import pythae

# Note - you must have torchvision installed for this example
from torch.utils.data import DataLoader
from bio_vae.models.legacy import vq_vae

from bio_vae.utils import collate_none, get_test_image
from bio_vae.datasets import BroadDataset, DatasetGlob
from bio_vae.shapes.transforms import (
    CropCentroidPipeline,
    MaskToDistogramPipeline,
    DistogramToMaskPipeline,
)
from bio_vae.shapes.transforms import (
    DistogramToCoords,
)

from bio_vae.models import VQ_VAE, Bio_VAE, VAE
from bio_vae.lightning import LitAutoEncoderTorch
from bio_vae.models.legacy import vae

interp_size = 128 * 4
window_size = 64
batch_size = 32

num_embeddings = 512
decay = 0.99
channels = 1
learning_rate = 1e-3

channels = 1

input_dim = (channels, window_size, window_size)
latent_dim = 64

datasets = [
    BroadDataset(
        "BBBC010",
        download=True,
        transform=MaskToDistogramPipeline(window_size, interp_size),
    ),
    BroadDataset(
        "BBBC010",
        download=True,
        transform=MaskToDistogramPipeline(
            window_size, interp_size, matrix_normalised=True
        ),
    ),
]


model_config_vae = pythae.models.VAEConfig(input_dim=input_dim, latent_dim=latent_dim)

model_config_vqvae = pythae.models.VQVAEConfig(
    input_dim=input_dim, latent_dim=latent_dim, num_embeddings=num_embeddings
)


models = [
    vq_vae.VQ_VAE(channels=channels),
    pythae.models.VQVAE(
        model_config_vqvae,
        encoder=vq_vae.Encoder(
            model_config_vqvae,
        ),
        decoder=vq_vae.Decoder(
            model_config_vqvae,
        ),
    ),
    pythae.models.VQVAE(
        model_config=model_config_vqvae,
    ),
    pythae.models.VAE(
        model_config=model_config_vae,
        encoder=vae.Encoder(model_config_vae),
        decoder=vae.Decoder(model_config_vae),
    ),
    pythae.models.VAE(
        model_config=model_config_vae,
    ),
]


@pytest.mark.parametrize("model", models)
class TestVAE:
    # def setup(self):

    # self.model = VAE(3, 10)
    # self.transform = MaskToDistogramPipeline(window_size, interp_size)
    # self.dataset = BroadDataset(
    #     "BBBC010", download=True, transform=transformer_dist
    # )

    # def test_summary(self):
    #     print(summary(self.model, (1, 64, 64), device='cpu'))
    #     # print(summary(self.model2, (3, 64, 64), device='cpu'))

    def test_forward(self, model):
        x = torch.randn(1, channels, window_size, window_size)
        y = model.forward({"data": x})
        print("Model Output size:", y[0].size())
        # print("Model2 Output size:", self.model2(x)[0].size())

    # Loss check buggy and inconsistent
    # def test_loss(self, model):
    # print("good")
    #     x = torch.randn(16, channels, 64, 64)

    #     result = model(x)
    #     loss = model.loss_function(*result, M_N=0.005)
    #     print(loss)


@pytest.mark.parametrize("dataset", datasets)
@pytest.mark.parametrize("model", models)
class TestMask:
    def test_mask_forward(self, model, dataset):
        model = Bio_VAE(model)
        test_img = get_test_image(dataset)
        z, log_var = model.encode(test_img)
        y_prime = model.decode(z)
        model.forward(test_img)

    def test_pipeline_forward(self, model, dataset):
        # dist = MaskToDistogramPipeline(window_size)(train_dataset_raw[0])
        # plt.imshow(dist)
        # plt.savefig("tests/test_mask_to_dist.png")
        # plt.close()
        # plt.close()
        dist = get_test_image(dataset)
        plt.imshow(dist.squeeze())
        plt.savefig("tests/test_pipeline_forward.png")
        plt.close()
        mask = DistogramToMaskPipeline(window_size)(dist)
        plt.imshow(mask.squeeze())
        plt.savefig("tests/test_dist_to_mask.png")
        plt.close()


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("dataset", datasets)
class TestModels:
    def test_dist_to_coord(self, model, dataset):
        # dist = transformer_dist(train_dataset[0][0])
        # TODO Faulty?
        test_img = get_test_image(dataset)
        coords = DistogramToCoords(window_size)(test_img)
        plt.scatter(coords[0][0][:, 0], coords[0][0][:, 1])
        plt.savefig("tests/test_dist_to_coord.png")
        plt.close()

    def test_dist_to_coord(self, model, dataset):
        test_img = get_test_image(dataset)
        # dist = transformer_dist(train_dataset[0][0])
        coords = DistogramToCoords(window_size)(test_img)
        plt.scatter(coords[0][:, 0], coords[0][:, 1])
        plt.savefig("tests/test_dist_to_coord.png")
        plt.close()

    def test_models(self, model, dataset):
        # vae = AutoEncoder(1, 1)
        # vae = VQ_VAE(channels=1)

        test_img = get_test_image(dataset)
        # loss, x_recon, perplexity = model(img)
        result = model(test_img)
        z, log_var = model.encode(test_img)
        y_prime = model.decode(z)
        # print(f"img_dims:{img.shape} y:_dims:{x_recon.shape}")
        print(f"img_dims:{test_img.shape}, z:_dims:{z.shape}")

    # @pytest.mark.skip(reason="Crashes github actions")
    @pytest.mark.parametrize("wrapper", [Bio_VAE])
    def test_mask_training(self, model, dataset, wrapper):
        model = wrapper(model)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_none,
        )
        lit_model = LitAutoEncoderTorch(model)
        trainer = pl.Trainer(
            max_steps=1,
            max_epochs=1,
        )  # .from_argparse_args(args)
        # trainer.test(lit_model, dataloader)
        trainer.fit(lit_model, dataloader)


# @pytest.mark.skipif(sys.version_info < (3,3))
# def test_model(model):
#     for i in range(10):
#         z_random = torch.normal(torch.zeros_like(z), torch.ones_like(z)).cuda()
#         generated_image = model.autoencoder.decoder(z_random)
#         plt.imshow(transforms.ToPILImage()(generated_image[0]))
#         plt.close()

# def test_mask_vae():
#     MaskVAE(VQ_VAE(channels=1))
