FROM continuumio/miniconda3
COPY . .
RUN conda env update -f environment.yml --name base
