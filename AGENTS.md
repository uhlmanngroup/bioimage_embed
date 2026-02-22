# bioimage_embed: Autoencoders for Biological Image Data

bioimage_embed is an all-in-one Python package designed to cater to the needs of computational biologists, data scientists, and researchers working on biological image data. With specialized…

## Context
- **Owner**: ctr26
- **Agent**: `science`
- **Focus**: research, data science, scientific computing
- **Tech**: python-package, conda, make, docker, notebooks, github-actions

## Key Files
- `README.md`
- `Makefile`
- `pyproject.toml`
- `config.yaml`

## Commands
```bash
make test  # poetry run pytest -v --tb=no
make figures  # python scripts/generate_figures.py
```

## Related
- `../Binomial-splitting/`
- `../agentic-prompt-engineering/`
- `../essays-published/`
- `../hooke/`

## Agent Guidance
- Prefer reproducible environments (conda/poetry) over ad-hoc installs
- Run `make test` (or equivalent) before committing
- Keep notebooks exploratory; move stable code to `src/`
- Document experiments in `memory/research/`

## Git Rules

### Before ANY push, pre-flight check:
```bash
# 1. Verify remote shows correct account
git remote -v | grep -E "(ctr26|craggles17)/" || echo "⚠️ WRONG ACCOUNT"

# 2. Test remote access
git ls-remote origin HEAD >/dev/null 2>&1 || echo "⚠️ CANNOT ACCESS REMOTE"

# 3. Only then push
git push ...
```

### Context before working:
```bash
git log --oneline -20                    # Recent history
git log --oneline --grep="<keyword>"     # Find relevant commits
git show <commit>                        # Understand changes
```

## Credentials

This repo uses ctr26 credentials.
Verify with: `gh auth status`

If wrong account, switch: `gh auth switch`
