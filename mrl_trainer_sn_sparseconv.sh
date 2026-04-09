# #!/bin/bash


# Interrompe o script se houver erro
set -e

# Definição da escala de imagens
# NUM_IMGS=(1 2 4 8)
NUM_IMGS=(1 2 4 8 10 12)

for n in "${NUM_IMGS[@]}"; do
    
    TRIAL="18marco_SparseConv_ShapeNet_n${n}_ehagoraDeus"
    
    echo "======================================================="
    echo "EXECUTANDO: Experimento com $n Imagens"
    echo "Trial Name: $TRIAL"
    echo "======================================================="

    # Execução direta no terminal
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29502 main.py \
        --config configs/Pre-Training/mrl_sparseconv.yaml \
        dataset.num_imgs=$n \
        --trial_name "$TRIAL"

    echo -e "\nFinalizado experimento para n=$n\n"
    
done

echo "Todos os 4 experimentos foram concluídos!"