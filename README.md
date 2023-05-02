# Optimizing Dynamic Neural Networks with Brainstorm

This is the artifact for the paper "Optimizing Dynamic Neural Networks with Brainstorm", We are going to reproduce the main results of the paper in this artifact.

> **NOTE:** For artifact evaluation committee, please directly jump to [Reproducing the Results](#reproducing-the-results) section and reproduce the results on the provided server. For other readers, please follow the instructions below to setup the environment and reproduce the results on your own server.

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

## Installation with Docker Container

### Building the Docker Image

We also provide a docker image to setup the environment. The docker image can be built by the following command:

```bash
python scripts/docker_gh_build.py --type latest
```

### Starting the Docker Container

The docker image can be run by the following command:

```bash
bash scripts/docker_run.sh
```

# Preparing the Dataset, Checkpoints, and Kernel Database:

```bash
bash scripts/init_dev.sh
```

# Reproducing the Results


Please enter the directory `scripts/artifact` and run the following commands to reproduce the results. Each scripts will run the corresponding experiment, visualize the results, and the figure will be saved in the `.cache/results/figures` directory. Therefore, firstly we need to enter the directory with the following command:

```bash
cd scripts/artifact
```

## Reproducing the Results on Single-GPU Server (~ 5 hours)

```bash
bash sing_gpu.sh
```

## Reproducing the Results on Multiple-GPU Server (~2 hours)

```bash
bash multi_gpu.sh
```
> **Note**: Due to the inability to access the machine with multiple GPUs inside the company and long evaluation time, we provide the screencasts for the results of multiple gpus. The table below shows our experimental results after screen recording.

<!-- | Single-GPU |    [single_gpu.mp4](https://drive.google.com/file/d/1ajXGo2wDrLfVioqH8iTuTVFmXS4OReRs/view?usp=share_link)    |           [Figures.tar.gz](https://drive.google.com/file/d/12-4z2sKjxfhl8FzVPbSlvw48J9JKiHWQ/view?usp=sharing)          |    `xxxxx`   | -->
|    Server    | Screencast | Visualized Results |  md5  |
| :--------: | :--------: | :----------------: | :---: |
| Multi-GPUs |[multi_gpu.mp4](https://drive.google.com/file/d/1ajXGo2wDrLfVioqH8iTuTVFmXS4OReRs/view?usp=share_link) | [Figures.tar.gz](https://drive.google.com/file/d/12-4z2sKjxfhl8FzVPbSlvw48J9JKiHWQ/view?usp=sharing) |`a1800694cb6a3c1f508cb57905dff498`|

> In the screeencast, we will first display the branch information of the code repository, then start the experiment using a one-click script. The script will delete the "results" path first. After running all experiments, it will enter the newly created "results" folder to compress the `figures` folder, calculate and output its corresponding md5 value. Reviewers can use this value to verify consistency between the compressed `figures` in the screen recording and that of provided one in above table.