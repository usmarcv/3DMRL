#!/bin/bash

set -e

NUM_IMGS=(2 8 12)

LRS=("5e-4")

BATCH_SIZES=(50)

# Loop sobre os batch sizes
for bs in "${BATCH_SIZES[@]}"; do

    # Loop sobre os números de imagens
    for n in "${NUM_IMGS[@]}"; do
        
        # Loop sobre os learning rates
        for lr in "${LRS[@]}"; do
            
            lr_str=$(echo "$lr" | tr -d '.')
            
            trial_name="ablation_3DMRL_PointBERT_bs${bs}_lr${lr_str}_sn_data_n${n}"
            
            echo "========================================================================"
            echo "Iniciando execução..."
            echo " - batch_size: $bs"
            echo " - num_imgs: $n"
            echo " - learning_rate: $lr"
            echo " - trial_name: $trial_name"
            echo "========================================================================"
            
            # Executa o comando repassando todos os parâmetros dinâmicos
            CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29001 main.py \
                --config configs/Pre-Training/mrl_pointbert_shapenet.yaml \
                dataset.num_imgs="$n" \
                training.lr="$lr" \
                batch_size="$bs" \
                --trial_name "$trial_name"
                
        done
    done
done

echo "========================================================================"
echo "Todos os treinamentos foram concluídos com sucesso!"