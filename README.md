bio_vae is a Python package that provides a convenient way to train and use autoencoders on biological image data. It includes functions for loading and preprocessing images from common sources such as microscopy, and tools for visualizing the results of the autoencoder's encoding and decoding process.

Installation
To install bio_vae, use pip:

Copy code
pip install bio_vae
Usage
Loading and Preprocessing Images
bio_vae includes functions for loading and preprocessing images from microscopy datasets, such as those in the BioImage Data Resource and Cell Image Library.

# Installation

    pip install git+https://github.com/ctr26/bio_vae
    
   

# Mask-VAE

This project attempts to using modern auto-encoding CNNs to access the latent shape space of a given dataset.
We demonstrate this on microscopy images of nuclei and C. Elegans (worms).
The project includes some sensible tricks such as including symmetry $\min|M^T - M|_2^2$ and $\min|diag(M)|$ constraints to help the model learn that it's using distance matrices.

## How it works

- Take image masks (white on black)
- Find their contour (thx [scikit image]((https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours)))
- Resample the contour to a standard length
- Create euclidean distance matrix
- Feed matrix as image into VAE
- Train model on distance matrix
- Opt. Convert distance matrix back to mask using MultiDimensionalScaling

Potential uses for this projects are:

- Synthetic shape generation for dataset augmentation
- Shape-based phenotyping in the latent space

## Usage

### Get data

    make download.data

If you don't have a Kaggle account you must create one and then follow the next steps:
1. Install the Kaggle API package so you can download the data from the Makefile you have all the information in their [Github repository](https://github.com/Kaggle/kaggle-api).
2. To use the Kaggle API you need also to create an API token. You can found how to do it in their [documentation](https://github.com/Kaggle/kaggle-api#api-credentials)
3. After that you will need to add your user and key in a file called `kaggle.json` in this location in your home directory `chmod 600 ~/.kaggle/kaggle.json`

### Intall

    poetry install

and or:

    pip install -e .

### Run

    python train.py

### TODO

- Scale invariant distance matrix encoding (scale by matrix norm)
- Find better sampling of contour, e.g. using Delaunay triangulation?
