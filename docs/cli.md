The cli is mostly handled by hydra (https://hydra.cc/docs/intro/). The main commands are:

bie_train: Train a model
bie_predict: Predict with a model

# Training

To train a model, you can use the following command:

```bash
bie_train
```

To see all the available options, you can use the `--help` flag:

```bash
bie_train --help
```

## Data

Out of the box bie_train is configured to try to use torchvision.datasets.ImageFolder to load data.
This can be endlessly overwritte using Hydra's configuration system (e.g. _target_ ).
However, for most applications using the stock ImageFolder class will work.
To then point the model to useful data you need to set the 'receipe.data' key like so:

```bash
bie_train recipe.data=/path/to/data
```

ImageFolder will use PIL to load images, so you can use any image format that PIL supports, this includes jpg, png, bmp, etc, tif.

More exotic formats will require a custom dataset class, which is not covered here; realisitically you should convert your data to a more common format.
PNG for instance is a lossless format that loads quickly from disk due to it's efficient compression.
The bie_train defaults tend to be sane, for instance the data is shuffled, and the data is split into train and validation sets.

It is worth noting that ImageFolder expects the data to be organised into "classes" even though default bie_train does not use the class labels during training.
To denote these classes, you should organise your data into folders, where each folder is a class, and the images in that folder are instances of that class.
See here for more information: https://pytorch.org/vision/stable/datasets.html#imagefolder

## Models

The default model backbone a "resnet18" with a "vae" architecture for autoencoding, but you can specify a different model using the `receipe.model` flag:

```bash
bie_train recipe.model=resnet50_vqvae receipe.data=/path/to/data
```

N.B. the resnet series of models expect the tensor input to (3,224,224) in shape,


### Supervised vs Unsupervised models

By default the model is unsupervised, meaning the class labels are ignored during training.
However, a (experimental) supervised model can be selected by setting:

```bash
bie_train lit_model.model=_target_="bioimage_embed.lightning.torch.AutoEncoderSupervised" receipe.data=/path/to/data
```

This uses contrastive learning using the labelled data, specifically SimCLR: https://arxiv.org/abs/2002.05709

## Reciepes

The major components of the training process are controlled by the "reciepe" schema.
These values are also what is used for generating the uuid of the training run.
This means that the model can infact resume from a crash or be retrained with the same configuration aswell as multiple models being trained in parallel using the same directory.
This is useful for hyperparameter search, or for training multiple models on the same data.

### lr_scheduler and optimizer

The lr_scheduler and optimizer are mimics of the timm library and built using create_optimizer and create_scheduler.
https://timm.fast.ai/Optimizers
and
https://timm.fast.ai/schedulerss

The default optimizer is "adamw" and the default scheduler is "cosine", aswell as some other hyperparameters borrowed from: https://arxiv.org/abs/2110.00476

The way the timm create_* functions work is they receive a generic SimpleNamespace, and only take the keys they need.
The consequence is that timm creates a controlled vocabulary for the hyperparameters in receipe; this makes it possible to choose from the wide variety of optimizers and schedulers in timm.
https://timm.fast.ai

## Augmentation

The package includes a default augmentation, which is stored in the configruation file.
The default augmentation is written using albumentations, which is a powerful library for image augmentation.
https://albumentations.ai/docs/


The default augmentation is a simple set of augmentations that are useful for biological_images, crucially it mostly neglects any RGB and non-physical augmentation effects.
It is recommended to edit the default augmentations in the configuration file and not in the CLI as the commands can get quite long.


## Config file

This will train a model using the default configuration. You can also specify a configuration file using the `--config` flag:

```bash
bie_train --config path/to/config.yaml
```
