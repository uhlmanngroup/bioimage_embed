#!make
include .env
include secrets.env
export

download.data:
	kaggle competitions download -c data-science-bowl-2018

test:
	pytest
	
all: get_data_list build

build:
	conda activate torch
	python idr_get_data.py

get_data_list:
	ls /nfs/bioimage/drop/idr*/**/*.tiff > file_list.txt
	ls -u /nfs/bioimage/drop/idr*/**/*.tiff > file_list.txt

run.on.cloud:
	python idr_get_data_s3.py

run.on.cloud.snake:
	snakemake --use-conda --cores all --verbose --google-lifesciences --default-remote-prefix idr-hipsci --google-lifesciences-region eu-west2

run.snake:
	snakemake  --cores all -F --use-conda --verbose

get.env.file:
	conda env export --from-history -f environment.yml -n torch