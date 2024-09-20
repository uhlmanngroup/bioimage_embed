#!make
include .env
include secrets.env
export

@PHONY: all test

test:
	poetry run pytest -v --tb=no
