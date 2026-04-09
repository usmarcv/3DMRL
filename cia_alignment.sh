#!/bin/bash
set -e  # para tudo se algum comando falhar

echo "Starting alignment training runs..."
# echo "Run 1: alignment with ShapeNet..."
torchrun --nproc_per_node=1 main_original.py \
  --config configs/Pre-Training/mrl_cia_ShapeNet.yaml \
  --trial_name MRL_CIA_ShapeNet_bs256 &

# echo "Run 2: alignment with Ensembled..."
#torchrun --nproc_per_node=1 main_original.py \
#  --config configs/Pre-Training/mrl_cia_Ensembled.yaml \
#  --trial_name MRL_CIA_Ensembled_bs256 &

# echo "Run 3: alignment with NoLVIS..."
#torchrun --nproc_per_node=1 main_original.py \
#  --config configs/Pre-Training/mrl_cia_NoLVIS.yaml \
#  --trial_name MRL_CIA_NoLVIS_bs256 &

#wait
echo "All alignment trainings completed!"
