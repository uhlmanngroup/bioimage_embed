#  %%
import bioimage_embed
import torch
import matplotlib.pyplot as plt


#  %%

# Note - you must have torchvision installed for this example
import os
import matplotlib.pyplot as plt
import torch

from bioimage_embed.shapes.transforms import (
    DistogramToCoords,
    CropCentroidPipeline,
)
from bioimage_embed.shapes.transforms import (
    DistogramToCoords,
    MaskToDistogramPipeline,
    AsymmetricDistogramToMaskPipeline,
)


interp_size = 128 * 4
window_size = 128 - 32

# train_dataset_glob = "data/stage1_train/*/masks/*.png"
# train_dataset_glob = "data/BBBC010_v1_foreground_eachworm/*.png"

# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")

transformer_crop = CropCentroidPipeline(window_size)
transformer_dist = MaskToDistogramPipeline(window_size, interp_size)
transformer_coords = DistogramToCoords(window_size)

# train_dataset_raw = DatasetGlob(train_dataset_glob)
# train_dataset_crop = DatasetGlob(
#     train_dataset_glob, transform=CropCentroidPipeline(window_size))
# train_dataset = DatasetGlob(train_dataset_glob, transform=transformer_dist)
# train_dataset_crop = DatasetGlob(train_dataset_glob, transform=transformer_crop)
train_dataset = None
transformer_dist = None
train_dataset_crop = None
# transformer_crop)


# img_squeeze = train_dataset_crop[0].unsqueeze(0)
# img_crop = train_dataset_crop[0]


# test_img = train_dataset[1].unsqueeze(0)
# # img_squeeze = train_dataset_crop[1].unsqueeze(0)
# img_crop = train_dataset_crop[1].unsqueeze(0)

# train_dataset = train_dataset_dist

input_dim = (3, window_size, window_size)
latent_dim = 16
model = bioimage_embed.models.create_model(
    "resnet18_vqvae_legacy", input_dim, latent_dim, pretrained=False, progress=False
)


def test_img_test():
    plt.close()
    plt.imshow(test_img[0][0])
    plt.close()


def test_forward_test():
    plt.close()
    results = model.forward(test_img)
    # y_prime = y_prime.detach().numpy()
    # plt.imshow(y_prime[0][0])
    # plt.close()


def test_encoder_decoder():
    plt.close()
    z = model.encoder(test_img)
    y_prime = model.decoder(z).detach().numpy()
    plt.imshow(y_prime[0][0])
    plt.close()


def test_generate():
    z = model.encoder(test_img)
    z_random = torch.normal(torch.zeros_like(z), torch.ones_like(z))
    # z_random = torch.ones_like(z)
    z_random = z + torch.normal(torch.zeros_like(z), torch.ones_like(z)) / 20
    generated_image_dist = model.decoder(z_random).detach().numpy()
    # dist_stack = np.stack(
    #     [generated_image_dist[0], np.transpose(generated_image_dist[0],)], axis=0)
    # dist_stack = np.stack(
    #     [generated_image_dist, generated_image_dist.transpose(0,1, 3, 2)], axis=0)

    # symmetric_generated_image_dist = np.max(dist_stack, axis=0)
    # print(symmetric_generated_image_dist.shape)
    # plt.imshow(symmetric_generated_image_dist[0][0])
    # plt.savefig("test_dist.png")
    # plt.close()

    # plt.imshow(symmetric_generated_image_dist[0][0])
    # plt.savefig("test_dist.png")
    # plt.close()

    # coords = DistogramToCoords(window_size)(symmetric_generated_image_dist[0])
    # plt.scatter(coords[0, :, 0], coords[0, :, 1])
    # plt.close()
    mask = AsymmetricDistogramToMaskPipeline(window_size)(generated_image_dist)

    # generated_image = model.model.mask_from_latent(z_random,window_size)
    # plt.imshow((mask[0]))
    # plt.savefig("test_dist_mask.png")
    # plt.close()

    # plt.imshow(train_dataset_crop[0][0])
    # plt.close()
    plt.close()
    plt.imshow(img_crop[0][0])
    plt.savefig("tests/test_generate_img_crop.png")
    plt.close()
    plt.imshow(mask[0][0])
    plt.savefig("tests/test_generate_mask.png")
    plt.close()


# %%
