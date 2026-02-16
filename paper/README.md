# bioimage_embed Paper

This directory contains the LaTeX source for the bioimage_embed paper.

## Title

**bioimage_embed: A Modular Framework for Self-Supervised Learning of Biological Image Embeddings**

## Target Venues

- Nature Methods (Applications Note)
- Bioinformatics
- Journal of Open Source Software (JOSS)

## Structure

```
paper/
├── main.tex              # Main document
├── sections/             # Individual sections
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── methods.tex
│   ├── experiments.tex
│   ├── results.tex
│   ├── discussion.tex
│   └── conclusion.tex
├── figures/              # Figure files
├── tables/               # Table files
├── references.bib        # Bibliography
├── Makefile             # Build automation
└── README.md            # This file
```

## Building

### Requirements

- LaTeX distribution (TeXLive, MiKTeX, or MacTeX)
- pdflatex
- bibtex

### Quick Start

```bash
# Build the paper
make

# Build and view
make view

# Clean build artifacts
make clean

# Quick compile (no bibtex)
make quick

# Continuous compilation
make watch
```

## TODOs

### Figures Needed

- [ ] Figure 1: Framework architecture diagram
- [ ] Figure 2: Reconstruction examples (input vs. output)
- [ ] Figure 3: t-SNE embeddings colored by compound/phenotype
- [ ] Figure 4: Transfer learning results across datasets

### Tables Needed

- [ ] Table 1: Reconstruction metrics (MSE, SSIM, FID)
- [ ] Table 2: Downstream classification accuracy
- [ ] Table 3: Training/inference time comparison

### Experiments to Run

- [ ] Cell Painting benchmarks (reconstruction + classification)
- [ ] IDR dataset evaluations
- [ ] Transfer learning experiments
- [ ] Ablation studies (latent dim, augmentations, pretraining)
- [ ] Reproducibility verification

### Writing Tasks

- [ ] Fill in author names and affiliations
- [ ] Add experimental results to Results section
- [ ] Create figures and tables
- [ ] Add citations for related work
- [ ] Write acknowledgements
- [ ] Add data/code availability statements
- [ ] Proofread and edit

## Notes

Based on `PUBLICATION_REVIEW.md`, the paper positions bioimage_embed as:

1. **Domain-specific tool** for biological image representation learning
2. **Modular framework** supporting multiple architectures (VAE, VQVAE, MAE)
3. **Production-ready** with modern ML engineering practices
4. **Accessible** via CLI for non-programmers

Key contributions:
- Native support for microscopy formats
- Integrated reproducibility infrastructure
- HuggingFace ecosystem integration
- Benchmarks on standard bioimage datasets

## Contact

For questions about the paper, contact [your email].
