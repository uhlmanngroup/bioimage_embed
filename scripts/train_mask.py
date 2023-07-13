# %%

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, KFold, train_test_split
from sklearn.metrics import make_scorer
import pandas as pd
from sklearn import metrics
import matplotlib as mpl

from pathlib import Path
import umap
from torch.autograd import Variable
from types import SimpleNamespace

import numpy as np

#  %%
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch

from bio_vae import shapes
import bio_vae

# Note - you must have torchvision installed for this example

from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from bio_vae.lightning import DataModule

from torchvision import datasets
from bio_vae.shapes.transforms import (
    ImageToCoords,
    CropCentroidPipeline,
    DistogramToCoords,
    MaskToDistogramPipeline,
)

# from bio_vae.models import Mask_VAE, VQ_VAE, VAE
import matplotlib.pyplot as plt

from bio_vae.lightning import DataModule
import matplotlib

# matplotlib.use("TkAgg")
interp_size = 128 * 2

max_epochs = 100

window_size = 128 * 2


params = {
    "epochs": 100,
    "batch_size": 4,
    "num_workers": 2**4,
    # "window_size": 64*2,
    "num_workers": 1,
    "input_dim": (1, window_size, window_size),
    # "channels": 3,
    "latent_dim": 16,
    "num_embeddings": 16,
    "num_hiddens": 16,
    "num_residual_hiddens": 32,
    "num_residual_layers": 150,
    # "embedding_dim": 32,
    # "num_embeddings": 16,
    "commitment_cost": 0.25,
    "decay": 0.99,
}

optimizer_params = {
    "opt": "LAMB",
    "lr": 0.001,
    "weight_decay": 0.0001,
    "momentum": 0.9,
}

lr_scheduler_params = {
    "sched": "cosine",
    "min_lr": 1e-4,
    "warmup_epochs": 5,
    "warmup_lr": 1e-6,
    "cooldown_epochs": 10,
    "t_max": 50,
    "cycle_momentum": False,
}

# channels = 3


# input_dim = (params["channels"], params["window_size"], params["window_size"])
args = SimpleNamespace(**params, **optimizer_params, **lr_scheduler_params)

dataset_path = "bbbc010/BBBC010_v1_foreground_eachworm"
# dataset = "bbbc010"
model_name = "vqvae"

train_data_path = f"data/{dataset_path}"
metadata = lambda x: f"results/{x}"

model_dir = f"models/{dataset_path}_{model_name}"
# %%

transform_crop = CropCentroidPipeline(window_size)
transform_dist = MaskToDistogramPipeline(window_size, interp_size)
transform_mdscoords = DistogramToCoords(window_size)
transform_coords = ImageToCoords(window_size)

transform_mask_to_gray = transforms.Compose([transforms.Grayscale(1)])

transform_mask_to_crop = transforms.Compose(
    [
        # transforms.ToTensor(),
        transform_mask_to_gray,
        transform_crop,
    ]
)

transform_mask_to_dist = transforms.Compose(
    [
        transform_mask_to_crop,
        transform_dist,
    ]
)
transform_mask_to_coords = transforms.Compose(
    [
        transform_mask_to_crop,
        transform_coords,
    ]
)

# train_data = torchvision.datasets.ImageFolder(
# "/home/ctr26/gdrive/+projects/idr_autoencode_torch/data/bbbc010"
# )
# train_dataset_crop = DatasetGlob(
#     train_dataset_glob, transform=CropCentroidPipeline(window_size))
transforms_dict = {
    "none": transform_mask_to_gray,
    "transform_crop": transform_mask_to_crop,
    "transform_dist": transform_mask_to_dist,
    "transform_coords": transform_mask_to_coords,
}

train_data = {
    key: datasets.ImageFolder(train_data_path, transform=value)
    for key, value in transforms_dict.items()
}

for key, value in train_data.items():
    print(key, len(value))
    plt.imshow(train_data[key][0][0], cmap="gray")
    plt.imsave(metadata(f"{key}.png"), train_data[key][0][0], cmap="gray")
    # plt.show()
    plt.close()


# plt.scatter(*train_data["transform_coords"][0][0])
# plt.savefig(metadata(f"transform_coords.png"))
# plt.show()

# plt.imshow(train_data["transform_crop"][0][0], cmap="gray")
# plt.scatter(*train_data["transform_coords"][0][0],c=np.arange(interp_size), cmap='rainbow', s=1)
# plt.show()
# plt.savefig(metadata(f"transform_coords.png"))


# Retrieve the coordinates and cropped image
coords = train_data["transform_coords"][0][0]
crop_image = train_data["transform_crop"][0][0]

fig = plt.figure(frameon=True)
ax = plt.Axes(fig, [0, 0, 1, 1])
ax.set_axis_off()
fig.add_axes(ax)

# Display the cropped image using grayscale colormap
plt.imshow(crop_image, cmap="gray_r")

# Scatter plot with smaller point size
plt.scatter(*coords, c=np.arange(interp_size), cmap="rainbow", s=1)

# Save the plot as an image without border and coordinate axes
plt.savefig(metadata(f"transform_coords.png"), bbox_inches="tight", pad_inches=0)

# Close the plot
plt.close()

# img_squeeze = train_dataset[0].unsqueeze(0)
# %%


# def my_collate(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch)

transform = transforms.Compose([transform_mask_to_dist, transforms.ToTensor()])

dataset = datasets.ImageFolder(train_data_path, transform=transform)

dataloader = DataModule(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    # transform=transform,
)

# dataloader = DataLoader(train_dataset, batch_size=batch_size,
#                         shuffle=True, num_workers=2**4, pin_memory=True, collate_fn=my_collate)

model = bio_vae.models.create_model("resnet18_vqvae_legacy", **vars(args))

lit_model = shapes.MaskEmbedLatentAugment(model, args)
lit_model = shapes.MaskEmbed(model, args)
# model = Mask_VAE("VAE", 1, 64,
#                      #  hidden_dims=[32, 64],
#                      image_dims=(interp_size, interp_size))

# model = Mask_VAE(VAE())
# %%
# lit_model = LitAutoEncoderTorch(model)

dataloader.setup()
model.eval()
# %%


# model_name = model._get_name()
model_dir = f"my_models/{dataset_path}_{model._get_name()}_{lit_model._get_name()}"

tb_logger = pl_loggers.TensorBoardLogger(f"logs/")

Path(f"{model_dir}/").mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(dirpath=f"{model_dir}/", save_last=True)

trainer = pl.Trainer(
    logger=tb_logger,
    gradient_clip_val=0.5,
    enable_checkpointing=True,
    devices="auto",
    accelerator="gpu",
    accumulate_grad_batches=4,
    callbacks=[checkpoint_callback],
    min_epochs=50,
    max_epochs=args.epochs,
)  # .from_argparse_args(args)

# %%

try:
    trainer.fit(lit_model, datamodule=dataloader, ckpt_path=f"{model_dir}/last.ckpt")
except:
    trainer.fit(lit_model, datamodule=dataloader)

lit_model.eval()

validation = trainer.validate(lit_model, datamodule=dataloader)
# testing = trainer.test(lit_model, datamodule=dataloader)
example_input = Variable(torch.rand(1, *args.input_dim))

# torch.jit.save(lit_model.to_torchscript(), f"{model_dir}/model.pt")
# torch.onnx.export(lit_model, example_input, f"{model_dir}/model.onnx")

# %%
# Inference
# predict_dataloader = DataLoader(dataset, batch_size=1)


dataloader = DataModule(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    # transform=transform,
)
dataloader.setup()

predictions = trainer.predict(lit_model, datamodule=dataloader)
latent_space = torch.stack(
    [prediction.z.flatten() for prediction in predictions[:-1]], dim=0
)

idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}


y = np.array([int(data[-1]) for data in dataloader.predict_dataloader()])[:-1]

y_partial = y.copy()
indices = np.random.choice(y.size, int(0.3 * y.size), replace=False)

y_partial[indices] = -1

classes = np.array([idx_to_class[i] for i in y])


# y = torch.stack([data[-1] for data in dataloader.dataset[:-1], dim=0)
# y = torch.stack([prediction.y for prediction in predictions[:-1]], dim=0)
# umap_space = umap.UMAP().fit(latent_space, y=y)
# umap_space = umap.UMAP().fit_transform(latent_space.numpy(), y=y)
mapper = umap.UMAP().fit(latent_space.numpy(), y=y)
semi_supervised_latent = mapper.transform(latent_space.numpy())
import umap.plot

umap.plot.points(mapper, labels=classes)
plt.savefig(metadata(f"umap.png"))
plt.savefig(metadata(f"umap.pgf"))
plt.show()
# fig, ax = plt.subplots(1, figsize=(14, 10))
# plt.scatter(*mapper.embedding_.T, s=5, c=y, cmap='Spectral', alpha=1.0)
# plt.setp(ax, xticks=[], yticks=[])
# cbar = plt.colorbar(boundaries=np.arange(3)-0.5)
# cbar.set_ticks(np.arange(2))
# cbar.set_ticklabels(set(classes))

# %%

X = latent_space.numpy()
y = classes


def scoring_df(X, y):
    # X = semi_supervised_latent
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Define a dictionary of metrics
    scoring = {
        "accuracy": make_scorer(metrics.accuracy_score),
        "precision": make_scorer(metrics.precision_score, average="macro"),
        "recall": make_scorer(metrics.recall_score, average="macro"),
        "f1": make_scorer(metrics.f1_score, average="macro"),
    }

    # Create a random forest classifier
    clf = RandomForestClassifier()

    # Specify the number of folds
    k_folds = 5

    # Perform k-fold cross-validation
    cv_results = cross_validate(
        estimator=clf,
        X=X,
        y=y,
        cv=KFold(n_splits=k_folds),
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    # Put the results into a DataFrame
    cv_results_df = pd.DataFrame(cv_results)
    return cv_results_df


mask_embed_score_df = scoring_df(X, y)
# Print the DataFrame
print(mask_embed_score_df)
mask_embed_score_df.to_csv(metadata(f"mask_embed_score_df.csv"))
