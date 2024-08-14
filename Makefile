#!make
include .env
include secrets.env
export

@PHONY: all download test

download.data:
	kaggle competitions download -c data-science-bowl-2018

test:
	poetry run pytest -v --tb=no
