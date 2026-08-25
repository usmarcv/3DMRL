# 3D-MRL: Nested Multimodal 3D Representations via Matryoshka Representation Learning
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

## Pre-training 

Run the training by the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29001 main.py --config configs/Pre-Training/mrl_pointbert_ensembled.yaml --trial_name 3dmlr_pointbert_ensembled

```

The configs can be found in `confis/Pre-Training` folder. You can also change the setting by passing arguments on the `.yaml's` file. You can find the runned models on the `exp` folder.


## Inference

Run the zero-shot evaluation by the following command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29001 test.py --config configs/Pre-Training/mrl_pointbert_shapenet.yaml --resume exp/path/to/checkpoint.pth
```

## Acknowledgement

3D-MRL is built using the awesome [ULIP](https://github.com/salesforce/ULIP), [OpenShape](https://github.com/Colin97/OpenShape_code), [Uni3D](https://github.com/baaivision/Uni3D), [TAMM](https://github.com/alanzhangcs/Tamm_Code) and [DuoDuo CLIP](https://github.com/3dlg-hcvc/DuoduoCLIP).

This work was supported in part by the São Paulo Research Foundation (FAPESP), under grants \#2024/09462-1 and \#2026/01721-3. The authors gratefully acknowledge the Center for Mathematical Sciences Applied to Industry (CeMEAI) for providing computational resources, funded by FAPESP under grant \#2013/07375-0.


## Citation
```bib
@article{lobo2026_3dmrl,
        author    = {Lobo, Márcus and Matias, Vitor and Farias, Jeová and Ponti, Moacir},
        title     = {3D-MRL: Nested Multimodal 3D Representations via Matryoshka Representation Learning},
        journal   = {BMVC},
        year      = {2026},
        }
```