FROM condaforge/mambaforge:22.9.0-3
COPY environment.yml .
COPY . .
RUN mamba env update -f environment.yml --name base
