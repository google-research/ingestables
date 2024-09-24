# A Dockerfile for training with PyTorch on GCP.

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
RUN apt-get update
RUN apt-get install -y python3-pip
RUN python3 -m pip install --upgrade pip --no-cache-dir

COPY ./torch /ingestables/torch
COPY ./README.md /README.md
COPY ./LICENSE /LICENSE
COPY ./pyproject.toml /pyproject.toml
WORKDIR /

RUN python3 -m pip install -e .

ENTRYPOINT ["python3", "/ingestables/torch/main.py"]
