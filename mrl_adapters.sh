#!/bin/bash

set -e

# Configurações de Hardware
GPUS=0,1,2,3
NUM_PROC=4
PORT=29502

# Valores para testar
BATCH_SIZES=(50)
RATIOS=(0.2 0.4)

# Caminho base
CONFIG_BASE="configs/Pre-Training/mrltamm.yaml"

for BATCH in "${BATCH_SIZES[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        
        TRIAL_NAME="07_abril_Ensembled_AdaptersMRL_bs_${BATCH}_ratio_${RATIO}_n1_newlr"
        
        echo "----------------------------------------------------------"
        echo "Iniciando Treino: $TRIAL_NAME"
        echo "Batch Size: $BATCH | Ratio: $RATIO"
        echo "----------------------------------------------------------"

        # Execução com overrides de argumentos
        # Nota: Assumindo que seu main_mrltamm.py aceita --batch_size e --ratio 
        # para sobrescrever o YAML. Se não aceitar, precisaremos editar o YAML via 'yq'.
        
        CUDA_VISIBLE_DEVICES=$GPUS torchrun \
            --nproc_per_node=$NUM_PROC \
            --master_port=$PORT \
            main_mrltamm.py \
            --config "$CONFIG_BASE" \
            --trial_name "$TRIAL_NAME" \
            batch_size=$BATCH \
            model.ratio=$RATIO

        echo "Finalizado: $TRIAL_NAME"
        sleep 5 # Pequena pausa para limpeza de memória da GPU
    done
done