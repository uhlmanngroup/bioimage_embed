#  %%
import matplotlib.pyplot as plt
import numpy as np
import pythae
import torch
import umap
import umap.plot
from pythae.models import VAE, VAEConfig
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
#  %%
from torch.utils.data import DataLoader, Dataset
# Note - you must have torchvision installed for this example
from torchvision import transforms
from tqdm import tqdm

from bio_vae.datasets import DatasetGlob
from bio_vae.lightning import LitAutoEncoderTorch
from bio_vae.models import Bio_VAE

latent_dim = 64
window_size = 64 * 2

batch_size = 32
num_training_updates = 15000

num_hiddens = 64
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3
dataset = "ivy_gap"
data_dir = "data"
channels = 3

input_dim = (channels, window_size, window_size)
model_name = "VQ_VAE"
train_dataset_glob = f"{data_dir}/{dataset}/random/*png"
model_dir = f"models/{dataset}_{model_name}"
ckpt_file = "models/ivy_gap_VQ_VAE/last.ckpt"

model_config_vqvae = pythae.models.VQVAEConfig(
    input_dim=input_dim, latent_dim=latent_dim, num_embeddings=num_embeddings
)
model = Bio_VAE("VQ_VAE", model_config=model_config_vqvae, channels=channels)
# model = Mask_VAE(VAE(1, 64, image_dims=(interp_size, interp_size)))

lit_model = LitAutoEncoderTorch(model)
model = LitAutoEncoderTorch(model).load_from_checkpoint(ckpt_file, model=model)
train_dataset = DatasetGlob(train_dataset_glob)

train_dataset = DatasetGlob(train_dataset_glob, transform=transforms.ToTensor())

# %%
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(4.0, 4.0))
grid = ImageGrid(
    fig,
    111,  # similar to subplot(111)
    nrows_ncols=(4, 4),  # creates 2x2 grid of axes
    axes_pad=0.1,  # pad between axes in inch.
)

for i, ax in enumerate(grid):
    ax.imshow(train_dataset[i][0])
plt.show()
plt.close()
# %%


image_index = 5
test_img_in = train_dataset[image_index].unsqueeze(0)
z, log_var = model.encode(test_img_in)
z_np = z.detach().numpy()


axd = plt.figure(constrained_layout=True, figsize=(12, 8)).subplot_mosaic(
    """
    Aa
    Bb
    Cc
    """,
)

axd["A"].imshow(test_img_in[0][0])
axd["A"].set_title("test_img_in")

# axd["a"].imshow(test_img_out[0][0])
# axd["a"].set_title("test_img_out")


plt.show()
plt.close()
# %%
z_list = []
z_dict = {}
for i, data in enumerate(tqdm(train_dataset)):
    if data is not None:
        z, mu = model.encode(data.unsqueeze(0))
        z_list.append(z)
        z_dict[i] = z
    if len(z_list) >= 100:
        break
latent = torch.stack(z_list).detach().numpy()
#  %%

latent_umap = latent.reshape(latent.shape[0], -1)
unfit_umap = umap.UMAP(n_neighbors=3, min_dist=0.1, metric="cosine", random_state=42)
unfit_umap = umap.UMAP(random_state=42)

fit_umap = unfit_umap.fit(latent_umap)
proj = fit_umap.transform(latent_umap)

umap_z = fit_umap.transform(z.detach().numpy().reshape((1, latent_umap[0].shape[0])))

umap.plot.points(fit_umap)
plt.show()
plt.close()
plt.scatter(proj[:, 0], proj[:, 1])
plt.savefig("latent_space.pdf")
plt.show()
plt.close()
# %%
# Sorted images


fig = plt.figure(figsize=(4.0, 4.0))
grid = ImageGrid(
    fig,
    111,  # similar to subplot(111)
    nrows_ncols=(10, 10),  # creates 2x2 grid of axes
    axes_pad=0.1,  # pad between axes in inch.
)

indices = np.argsort(proj[:, 0])

for i, ax in enumerate(grid):
    ax.imshow(train_dataset[indices[i]][0])
plt.show()
plt.close()

#  %% Principle worm components, probably meaningless

# pc_1 = fit_umap.inverse_transform(np.array([[1, 0]]))
# pc_2 = fit_umap.inverse_transform(np.array([[0, 1]]))

# plt.plot(pc_1.T)
# plt.show()
# plt.close()
# plt.plot(pc_2.T)
# plt.show()
# plt.close()
# # %%
# coord_1 = np.array([[0, -1]])
# coord_2 = np.array([[-1, 10]])

# coord_1 = np.array([[5, 4]])
# coord_2 = np.array([[-13.8, 7.5]])
# coord_2 = np.array([[1, 7.5]])

# plt.scatter(proj[:, 0], proj[:, 1])
# plt.scatter(coord_1[:, 0], coord_1[:, 1], label="coord_1")
# plt.scatter(coord_2[:, 0], coord_2[:, 1], label="coord_2")
# plt.legend()
# plt.save_fig("latent_with_coord.pdf")
# plt.show()

# %%
# from scipy.stats import gaussian_kde

# z = gaussian_kde(proj.T)(proj.T)

# plt.scatter(proj[:, 0], proj[:, 1], c=gaussian_kde(proj.T)(proj.T))
# plt.scatter(coord_1[:, 0], coord_1[:, 1], label="Sample 1")
# plt.scatter(coord_2[:, 0], coord_2[:, 1], label="Sample 2")
# plt.legend()
# plt.show()


# %%

# from scipy.spatial import distance


# def closest_node(node, nodes):
#     closest_index = distance.cdist([node], nodes).argmin()
#     return nodes[closest_index], closest_index


# real_worm_near_1_coord, real_worm_near_1_index = closest_node(coord_1[0], proj)
# real_worm_near_2_coord, real_worm_near_2_index = closest_node(coord_2[0], proj)


# real_worm_1 = train_dataset_crop[real_worm_near_1_index].unsqueeze(0)
# plt.imshow(real_worm_1[0][0], cmap="binary")
# plt.xlim(0.25 * 500, 0.75 * 500)
# plt.ylim(0.25 * 500, 0.75 * 500)
# plt.show()

# real_worm_2 = train_dataset_crop[real_worm_near_2_index].unsqueeze(0)
# plt.imshow(real_worm_2[0][0], cmap="binary")
# plt.xlim(0.25 * 500, 0.75 * 500)
# plt.ylim(0.25 * 500, 0.75 * 500)
# plt.show()

# %%


# %%