# Optimizing Dynamic Neural Networks with Brainstorm

This is the artifact for the paper "Optimizing Dynamic Neural Networks with Brainstorm", We are going to reproduce the main results of the paper in this artifact.

# Hardware and Software Requirements

- Hardware Requirements
  - Server with Signle GPU
    - CPU: AMD-EPYC-7V13 CPUs
    - GPU: NVIDIA A100 (80GB) GPU

  - Server with Multiple GPUs
    - CPU: Intel Xeon CPUE5-2690 v4 CPU
    - GPUs: NVIDIA V100 (32GB) GPUs x 8

- Software Requirements
  - CUDA 11.3
  - cuDNN 8.6 for NVIDIA A100 GPU and cuDNN 8.2 for NVIDIA V100 GPUs
  - [Pytorch 1.12.1 with CUDA 11.3](https://pytorch.org/get-started/previous-versions/#v1121)

# Setting up the Environment

We provide two options to set up the environment: bare metal server and docker container.

## Installation on Bare Metal Server

We provide an one-click script to setup the environment on bare metal server. The script will install the required packages and Brainstorm itself.

```bash
bash scripts/setup_bare.sh
```

## Building the Docker Image

We also provide a docker image to setup the environment. The docker image can be built by the following command:

```bash
python scripts/docker_gh_build.py --type latest
```

