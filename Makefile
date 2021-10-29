#!make
include .env
include secrets.env
export

<<<<<<< HEAD
download.data:
	kaggle competitions download -c data-science-bowl-2018

test:
	pytest
	
=======

GOOGLE_APPLICATION_CREDENTIALS=$(shell pwd)/credentials.json
BUCKET_NAME=idr-hipsci
TRAINING_DIR=idr0034-kilpinen-hipsci
PROJECT=prj-ext-dev-bia-binder-113155

JOB_PREFIX=vae
JOB_NAME=$(JOB_PREFIX)_$(shell date +%Y%m%d_%H%M%S)
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models
DATA_DIR=gs://${BUCKET_NAME}/${TRAINING_DIR}

.EXPORT_ALL_VARIABLES:
	GOOGLE_APPLICATION_CREDENTIALS
	BUCKET_NAME
	TRAINING_DIR
	JOB_PREFIX
	JOB_NAME
	JOB_DIR


# MY_VAR := $(shell echo whatever)

# test:
# 	@echo MY_VAR IS $(MY_VAR)

test:
	@echo $$GOOGLE_APPLICATION_CREDENTIALS $$BUCKET_NAME $$TRAINING_DIR

>>>>>>> 4f61d14 (adding gcp to makefile)
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
	snakemake --use-conda --cores all \
		--verbose --google-lifesciences \
		--default-remote-prefix idr-hipsci \
		--google-lifesciences-region eu-west2

run.snake:
	snakemake  --cores all -F --use-conda --verbose

get.env.file:
	conda env export --from-history -f environment.yml -n torch

on.gcp:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
	--region=europe-west2 \
	--master-image-uri=gcr.io/cloud-ml-public/training/pytorch-gpu.1-9 \
	--scale-tier=CUSTOM \
	--master-machine-type=n1-standard-8 \
	--master-accelerator=type=nvidia-tesla-t4,count=1 \
	--job-dir=${JOB_DIR} \
	--package-path=./trainer \
	--module-name=trainer.train \
	--stream-logs \
	-- \
	--num-epochs=10 \
	--batch-size=100 \
	--learning-rate=0.001 \
	--gpus=1


on.gcp.big:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
	--region=europe-west2 \
	--master-image-uri=gcr.io/cloud-ml-public/training/pytorch-gpu.1-9 \
	--config=config.yaml \
	--job-dir=${JOB_DIR} \
	--package-path=./trainer \
	--module-name=trainer.train \
	--stream-logs \
	-- \
	--num-epochs=10 \
	--batch-size=100 \
	--learning-rate=0.001 \
	--gpus=2 \
	--accelerator='ddp'\
	--num_nodes=3

tensorboard:
	tensorboard --logdir=gs://$(BUCKET_NAME)/${JOB_NAME}
download.data:
	kaggle competitions download -c data-science-bowl-2018

test:
	pytest
	