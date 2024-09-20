# bioimage_embed: Autoencoders for Biological Image Data

bioimage_embed is an all-in-one Python package designed to cater to the needs of computational biologists, data scientists, and researchers working on biological image data. With specialized functions to handle, preprocess, and visualize microscopy datasets, this tool is tailored to streamline the embedding process for biological imagery.

[![Build Status](https://img.shields.io/badge/build-passing-green.svg)](https://github.com/ctr26/bioimage_embed)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://github.com/ctr26/bioimage_embed)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/ctr26/bioimage_embed)

---

## Features

- Seamless loading of microscopy datasets, compatible with the BioImage Data Resource and Cell Image Library.
- Built-in preprocessing functions to ensure your images are primed for encoding.
- Visual tools to dive deep into the encoding and decoding processes of your autoencoders.

---

## Installation

To get started with bioimage_embed, you can install it directly via pip or from the GitHub repository.

### From PyPI:

```bash
pip install bioimage_embed
```

### From GitHub:

```bash
pip install git+https://github.com/ctr26/bioimage_embed
```

---

## Usage

### 1. Basic Installation:

```bash
pip install -e .
```

### 2. Command Line Interface (CLI):

To get a list of all commands and functions:

```bash
bioimage_embed --help
```

OR

```bash
bie --help
```


### 3. Developer Installation:

For those intending to contribute or looking for a deeper dive into the codebase, we use `poetry` to manage our dependencies and virtual environments:

```bash
poetry env use python
poetry install
poetry shell
```

---

## Support & Contribution

For any issues, please refer to our [issues page](https://github.com/ctr26/bioimage_embed/issues). Contributions are more than welcome! Please submit pull requests to the master branch.

---

## License

bioimage_embed is licensed under the MIT License. Please refer to the [LICENSE](https://github.com/ctr26/bioimage_embed/LICENSE) for more details.

---

Happy Embedding! 🧬🔬
