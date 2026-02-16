#!make
include .env
include secrets.env
export

@PHONY: all test figures

test:
	poetry run pytest -v --tb=no

figures:
	python scripts/generate_figures.py
