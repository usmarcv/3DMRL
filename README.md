# 3D Matryoshka Representation Learning for Multimodal 3D Understanding
This repository contains the official implementation of ''3D-MRL'' in our paper.

## Installation

If you would to run the inference or training locally, you may need to install the following dependencies:

- (i) Create a conda environment and install pytorch, MinkowskiEngine, and DGL by the following commands or their official guides:

```bash
conda create -n 3dmrl python=3.9
conda activate 3dmrl
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine
conda install -c dglteam/label/cu113 dgl
```

- (ii) Install the following packages:

```bash
pip install huggingface_hub wandb omegaconf torch_redstone einops tqdm open3d 
```

## Dataset Downloading

First, modify the path in the download_data.py. Then, execute the following command to download data from Hugging Face:

```bash
python3 download_data.py
```

The datasets used for experiments are the same as [OpenShape](https://github.com/Colin97/OpenShape_code). Please refer to OpenShape for more details of the data.

