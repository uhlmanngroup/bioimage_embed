# CLAUDE.md

**⚠️ Read this file FIRST before making any changes.**

**Parent Context**: See `~/projects/ctr26/research/CLAUDE.md` for research domain rules and `~/projects/craggles/AGENTS.md` for workspace orchestration.

## Project
Autoencoders for biological image data — all-in-one Python package for embedding microscopy datasets using variational autoencoders and self-supervised learning (pytorch-lightning).

## Stack
- Python 3.9–3.10 | `poetry` (existing) → migrate to `uv` when feasible
- Framework: `pytorch-lightning`, `umap-learn`, `jsonargparse`
- Tests: `pytest` | Notebooks: `jupytext`
- CLI: `bioimage_embed` / `bie` / `bie_train` / `bie_infer`

## Commands
```bash
poetry install                  # Install deps (current toolchain)
poetry run pytest -v --tb=no    # Run tests
poetry run python main.py       # Run training entrypoint
bioimage_embed --help           # CLI help
bie_train --help                # Train CLI
bie_infer --help                # Inference CLI
make figures                    # Regenerate figures
```

> **Toolchain migration target**: `uv` + `ruff` + `ty` (Python 3.11+). Until migrated, use poetry.

## Standards
- Type hints on all function signatures; `from __future__ import annotations` at top of every file
- Pydantic dataclasses over `BaseModel`; composition over inheritance
- No class hierarchies deeper than 1 level — prefer `Protocol` for polymorphism
- Terse one-line docstrings; no restating of type hints in prose
- Reproducible experiments: fix seeds, log to wandb/mlflow, parametrize configs via `jsonargparse`
- No notebook spaghetti — notebooks for EDA/figures only; logic lives in `src/`
- `pathlib.Path` over `os.path`; no mutable default args

## Git
- Branch: `feature/issue-N-desc` or `fix/issue-N-desc`
- Commits: conventional (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`)
- PRs required — no direct push to `main`
- `make test` must pass before opening PR

## Relevant Skills

**See**: `~/projects/craggles/memory/skill-index.md` for full descriptions

### For This Project
- **research-engineering-review** - Use for PR review checklist (#research)
- **parallel-git** - Multi-agent workflows if needed (#research)
- **biopython** - If working with biological sequences (#research)
- **anndata** - For single-cell data formats (related) (#research)
- **exploratory-data-analysis** - Initial data exploration (#research)
- **dask** - If parallelizing large image datasets (#research)
- **citation-management** - For paper references (#research)
- **llm-budget** - Manage API costs during development (#meta)

**When to use**: Check skill-index.md before starting new features. For microscopy-specific ML, combine pytorch with exploratory-data-analysis patterns.

## Review (mandatory)
- **Principle Engineer** reviews code quality, types, test coverage
- **Scientist** reviews embedding quality, autoencoder rigor, biological validity of results
- No merge without both approvals (see `AGENTS.md`)

## Domain Context
- Inputs: microscopy images (BioImage Data Resource, Cell Image Library formats)
- Key methods: VAE, β-VAE, self-supervised contrastive learning, UMAP embedding
- Outputs: low-dimensional embeddings for downstream phenotyping / clustering
