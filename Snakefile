# config = 
gpus = 0
rule all:
    input:
        dir("runs")
rule idr_get_data:
    input:
        "idr_get_data.py"
    output:
        dir("runs")
    resources:
        nvidia_gpu=gpus
    conda:
        "environment.yml",
    shell:
        "python {input}"