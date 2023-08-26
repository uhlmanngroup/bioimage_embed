bioimage_embed is a Python package that provides a convenient way to train and use autoencoders on biological image data. It includes functions for loading and preprocessing images from common sources such as microscopy, and tools for visualizing the results of the autoencoder's encoding and decoding process.

Installation
To install bioimage_embed, use pip:

Copy code
pip install bioimage_embed
Usage
Loading and Preprocessing Images
bioimage_embed includes functions for loading and preprocessing images from microscopy datasets, such as those in the BioImage Data Resource and Cell Image Library.

# Installation

    pip install git+https://github.com/ctr26/bioimage_embed
    
   
## Usage

### Install:

    pip install -e .

### CLI

    bioimage_embed --help
    
    or

    bie --help
### Get data

    make download.data

### Dev intall

Uses poetry to manage dependencies and virtualenvs

    poetry env use python 

    poetry install

    poetry shell

