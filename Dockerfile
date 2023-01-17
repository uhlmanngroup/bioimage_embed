# FROM condaforge/mambaforge:22.9.0-3
# # FROM continuumio/miniconda3
# COPY environment.yml .
# COPY . .
# RUN mamba env update -f environment.yml --name base
# FROM condaforge/mambaforge:22.9.0-3
FROM continuumio/miniconda3
COPY environment.yml .
COPY . .
RUN conda env update -f environment.yml --name base
