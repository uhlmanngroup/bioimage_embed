# %%
from pathlib import Path
import umap
from torch.autograd import Variable
from types import SimpleNamespace

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
    CropCentroidPipeline,
    DistogramToCoords,
    DistogramToCoords,
    MaskToDistogramPipeline,
)

# from bio_vae.models import Mask_VAE, VQ_VAE, VAE
import matplotlib.pyplot as plt

from bio_vae.lightning import DataModule
import matplotlib

matplotlib.use("TkAgg")
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


# train_dataset_glob = "data-science-bowl-2018/stage1_train/*/masks/*.png"
# train_dataset_glob = "data/stage1_train/*/masks/*.png"
# train_dataset_glob = f"data/{dataset}/*.png"
# %%
# train_dataset_glob = os.path.join("data/BBBC010_v1_foreground_eachworm/*.png")


# train_dataset_glob = os.path.join("data/DatasetGlob/train/masks/*.tif")
# test_dataloader_glob=os.path.join(os.path.expanduser("~"),
# "data-science-bowl-2018/stage1_test/*/masks/*.png")

# model_dir = "test"
# model_dir = "BBBC010_v1_foreground_eachworm"
model_dir = f"models/{dataset_path}_{model_name}"
# %%

transform_crop = CropCentroidPipeline(window_size)
transform_dist = MaskToDistogramPipeline(window_size, interp_size)
transform_coords = DistogramToCoords(window_size)

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
        transform_mask_to_dist,
        # transform_coords,
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
    # "transform_coords": transform_mask_to_coords,
}

train_data = {
    key: datasets.ImageFolder(train_data_path, transform=value)
    for key, value in transforms_dict.items()
}

for key, value in train_data.items():
    print(key, len(value))
    plt.imshow(train_data[key][0][0], cmap="gray")
    plt.imsave(f"{key}.png", train_data[key][0][0], cmap="gray")
    # plt.show()
    plt.close()


# train_dataset = DatasetGlob(train_data, transform=transforms.ToTensor())
# plt.imshow(train_dataset[0][0], cmap="gray")
# plt.show()
# train_dataset = DatasetGlob(train_dataset_glob, transform=transformer_crop)
# plt.imshow(train_dataset[0][0], cmap="gray")
# plt.show()

# train_dataset = DatasetGlob(train_dataset_glob, transform=transformer_crop)
# plt.imshow(train_dataset[0][0], cmap="gray")
# plt.show()

# train_dataset = DatasetGlob(train_dataset_glob, transform=transformer_dist)
# plt.imshow(train_dataset[0][0], cmap="gray")
# plt.show()


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

model = bio_vae.models.create_model("resnet50_vqvae_legacy", **vars(args))

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


model_name = model._get_name()
model_dir = f"my_models/{dataset_path}_{model_name}"

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

import numpy as np

y = np.array([int(data[-1]) for data in dataloader.predict_dataloader()])[:-1]
classes = np.array([idx_to_class[i] for i in y]) 
# y = torch.stack([data[-1] for data in dataloader.dataset[:-1], dim=0)
# y = torch.stack([prediction.y for prediction in predictions[:-1]], dim=0)
# umap_space = umap.UMAP().fit(latent_space, y=y)
umap_space = umap.UMAP().fit_transform(latent_space.numpy(), y=y)
mapper = umap.UMAP().fit(latent_space.numpy(), y=y)

import umap.plot
umap.plot.points(mapper, labels=classes)


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
X = latent_space.numpy()
y = classes
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
clf = RandomForestClassifier()

# Fit the model to the data
clf.fit(X_train, y_train)

# Use the model to make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate metric scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# %%

