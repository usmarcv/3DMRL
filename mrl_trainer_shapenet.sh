#!/bin/bash

set -e

# Define os parâmetros de imagens
NUM_IMGS=(1 4)

# Define os Learning Rates (substitua pelos dois valores desejados)
LRS=("1e-3" "5e-4")

# Define os caminhos dos checkpoints
PATHS=(
    "exp/07_abril_ShapeNet_AdaptersMRL_bs_50_ratio_0.2_n1_newlr@20260407-135857/ckpt/epoch_176.pt"
    "exp/07_abril_ShapeNet_AdaptersMRL_bs_50_ratio_0.4_n1_newlr@20260407-143727/ckpt/epoch_176.pt"
     # <- Preencha o path correspondente ao ratio 0.4
    # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0_n1_textcaption@20260403-152554/ckpt/epoch_194.pt"
    # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.2_n1_textcaption@20260403-160418/ckpt/epoch_199.pt"
    # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.4_n1_textcaption@20260403-164256/ckpt/epoch_192.pt"
    # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.6_n1_textcaption@20260403-172133/ckpt/epoch_199.pt"
    # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.8_n1_textcaption@20260403-180005/ckpt/epoch_199.pt"
    # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_1.0_n1_textcaption@20260403-183813/ckpt/epoch_178.pt"
)

# Define as taxas correspondentes a cada caminho acima
RATIOS=(0.2 0.4)

# Loop sobre os números de imagens (1 e 4)
for n in "${NUM_IMGS[@]}"; do
    
    # Loop sobre os learning rates
    for lr in "${LRS[@]}"; do
        
        # Loop sobre os índices dos arrays de path/ratio
        for i in "${!PATHS[@]}"; do
            path="${PATHS[$i]}"
            ratio="${RATIOS[$i]}"
            
            # Formatação do nome do trial: remove o ponto do ratio para nomes mais limpos (ex: 0.6 -> 06)
            ratio_str=$(echo "$ratio" | tr -d '.')
            
            # Formatação do LR para o nome do trial (remove pontos se usar notação decimal, ex: 0.001 -> 0001)
            lr_str=$(echo "$lr" | tr -d '.')
            
            # Se for exatamente '0', formata como '00'
            if [ "$ratio_str" == "0" ]; then 
                ratio_str="00"
            fi
            
            # Constrói o trial_name dinâmico incluindo o num_imgs, ratio e learning rate
            trial_name="07abril_AMRL2_bs50_r${ratio_str}_lr${lr_str}_sn_pb_n${n}"
            
            echo "========================================================================"
            echo "Iniciando execução..."
            echo " - num_imgs: $n"
            echo " - learning_rate: $lr"
            echo " - ratio: $ratio"
            echo " - trial_name: $trial_name"
            echo "========================================================================"
            
            # Executa o comando repassando todos os parâmetros
            CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29502 main_mrltamm.py \
                --config configs/Pre-Training/mrltamm2.yaml \
                dataset.num_imgs="$n" \
                training.lr="$lr" \
                --trial_name "$trial_name" \
                pretrained_adapters.path="$path" \
                pretrained_adapters.ratio="$ratio"
                
        done
    done
done

echo "========================================================================"
echo "Todos os treinamentos foram concluídos com sucesso!"

# set -e

# # Define os parâmetros de imagens
# NUM_IMGS=(1 4)

# # Define os caminhos dos checkpoints
# PATHS=(
#     "exp/07_abril_ShapeNet_AdaptersMRL_bs_50_ratio_0.2_n1_newlr@20260407-135857/ckpt/epoch_176.pt"
#     ""
#     # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0_n1_textcaption@20260403-152554/ckpt/epoch_194.pt"
#     # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.2_n1_textcaption@20260403-160418/ckpt/epoch_199.pt"
#     # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.4_n1_textcaption@20260403-164256/ckpt/epoch_192.pt"
#     # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.6_n1_textcaption@20260403-172133/ckpt/epoch_199.pt"
#     # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.8_n1_textcaption@20260403-180005/ckpt/epoch_199.pt"
#     # "exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_1.0_n1_textcaption@20260403-183813/ckpt/epoch_178.pt"
# )

# # Define as taxas correspondentes a cada caminho acima
# RATIOS=(0.2 0.4)

# # Loop sobre os números de imagens (1 e 4)
# for n in "${NUM_IMGS[@]}"; do
    
#     # Loop sobre os índices dos arrays de path/ratio
#     for i in "${!PATHS[@]}"; do
#         path="${PATHS[$i]}"
#         ratio="${RATIOS[$i]}"
        
#         # Formatação do nome do trial: remove o ponto do ratio para nomes mais limpos (ex: 0.6 -> 06)
#         ratio_str=$(echo "$ratio" | tr -d '.')
        
#         # Se for exatamente '0', formata como '00' (apenas para padronizar visualmente o nome)
#         if [ "$ratio_str" == "0" ]; then 
#             ratio_str="00"
#         fi
        
#         # Constrói o trial_name dinâmico com base nos exemplos
#         trial_name="07abril_AMRL2_bs50_r${ratio_str}_sn_pb_n${n}"
        
#         echo "========================================================================"
#         echo "Iniciando execução..."
#         echo " - num_imgs: $n"
#         echo " - ratio: $ratio"
#         echo " - trial_name: $trial_name"
#         echo "========================================================================"
        
#         # Executa o comando repassando todos os parâmetros concatenados
#         CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29502 main_mrltamm.py \
#             --config configs/Pre-Training/mrltamm2.yaml \
#             dataset.num_imgs="$n" \
#             --trial_name "$trial_name" \
#             pretrained_adapters.path="$path" \
#             pretrained_adapters.ratio="$ratio"
            
#     done
# done

# echo "========================================================================"
# echo "Todos os treinamentos foram concluídos com sucesso!"

# set -e

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_mrltamm.py --config configs/Pre-Training/mrltamm2.yaml dataset.num_imgs=4 --trial_name 03abril_teste_bs50_r01_sn_pb_n4

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main_mrltamm.py --config configs/Pre-Training/mrltamm2.yaml dataset.num_imgs=4 --trial_name 03abril_teste_bs50_r06_sn_pb_n4

# dataset.num_imgs: 1 4

# pretrained_adapters.path: 

# exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0_n1_textcaption@20260403-152554/ckpt/epoch_194.pt
# exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.2_n1_textcaption@20260403-160418/ckpt/epoch_199.pt
# exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.4_n1_textcaption@20260403-164256/ckpt/epoch_192.pt
# exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.6_n1_textcaption@20260403-172133/ckpt/epoch_199.pt
# exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_0.8_n1_textcaption@20260403-180005/ckpt/epoch_199.pt
# exp/03abril_AdaptersMRL_ShapeNet_bs50_ratio_1.0_n1_textcaption@20260403-183813/ckpt/epoch_178.pt

# pretrained_adapters.ratio: 0 0.2 0.4 0.6 0.8 1.0


# NUM_IMGS=(1 4)

# for n in "${NUM_IMGS[@]}"; do
    
#     TRIAL="30marco_MRL_Adapter_PB_n${n}_ratio06"
    
#     echo "======================================================="
#     echo "EXECUTANDO: Experimento com $n Imagens"
#     echo "Trial Name: $TRIAL"
#     echo "======================================================="
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29502 main_mrltamm.py \
#         --config configs/Pre-Training/mrltamm2.yaml \
#         dataset.num_imgs=$n \
#         --trial_name "$TRIAL"

#     echo -e "\nFinalizado experimento para n=$n\n"

# done

# echo "Todos os experimentos foram concluídos!"

# exp/18to19marco_PoinBERT_Ensembled_n4@20260318-205538/ckpt/epoch_30.pt
# # DIMS_OPTIONS=(
# #     "[10,20,40,80,160,320,640,1280]"
# #     "[20,40,80,160,320,640,1280]"
# #     "[40,80,160,320,640,1280]"
# #     "[80,160,320,640,1280]"
# #     "[160,320,640,1280]"
# #     "[320,640,1280]"
# #     "[640,1280]"
# #     "[1280]"
# # )

# # Configurações Fixas
# MODEL="PointBERT"
# LR=0.0005
# NUM_IMGS=1

# # Parâmetros de Variação
# BATCH_SIZES=(32 48 64 128) # Exemplo: testando dois tamanhos
# DIMS_OPTIONS=(
#     "[10,20,40,80,160,320,640,1280]"
#     "[20,40,80,160,320,640,1280]"
#     "[40,80,160,320,640,1280]"
#     "[80,160,320,640,1280]"
#     "[160,320,640,1280]"
#     "[320,640,1280]"
#     "[640,1280]"
#     "[1280]"
# )


# for bs in "${BATCH_SIZES[@]}"; do
#     for dims in "${DIMS_OPTIONS[@]}"; do
        
#         # Tratamento do nome das dimensões para a pasta
#         DIMS_NAME=$(echo $dims | sed 's/[[\]]//g' | sed 's/,/_/g')
        
#         # Nome do trial incluindo o Batch Size (BS)
#         TRIAL="13marco_MRL_Dims_${DIMS_NAME}_BS${bs}_LR${LR}"
        
#         echo "======================================================="
#         echo "EXECUTANDO: BS=$bs | Dims=$dims"
#         echo "Trial Name: $TRIAL"
#         echo "======================================================="

#         # Execução do treinamento
#         # Passamos batch_size para sobrescrever o valor do YAML
#         CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29001 main.py \
#             --config configs/Pre-Training/mrl_trainer_only_space.yaml \
#             model.name=$MODEL \
#             dataset.num_imgs=$NUM_IMGS \
#             dataset.train_batch_size=$bs \
#             training.lr=$LR \
#             mrl.use_mrl=True \
#             mrl.use_mrl_log_scale=False \
#             mrl.dims="$dims" \
#             --trial_name "$TRIAL"

#         echo -e "\nFinalizado: BS=$bs com Dims=$dims\n"
        
#     done
# done

# echo "Todos os experimentos da matriz de Batch Size e Dimensões foram concluídos!"


# echo "Matriz completa de experimentos concluída!"

# Interrompe o script se houver erro
# set -e

# # Parâmetros de entrada
# NUM_IMGS=(1 2 4 8)
# LOG_SCALE_OPTIONS=("True" "False")

# for n in "${NUM_IMGS[@]}"; do
#     for log_scale in "${LOG_SCALE_OPTIONS[@]}"; do
        
#         # Criando um trial_name único para diferenciar os resultados
#         TRIAL="PointBERT_MRL_ShapeNet_n${n}_LogScale_${log_scale}_AGORAVAI"
        
#         echo "======================================================="
#         echo "EXECUTANDO: Imagens=$n | LogScale=$log_scale"
#         echo "Trial Name: $TRIAL"
#         echo "======================================================="

#         # Comando de execução conforme sua solicitação
#         CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py \
#             --config configs/Pre-Training/mrl_pointbert.yaml \
#             dataset.num_imgs=$n \
#             training.use_mrl_log_scale=$log_scale \
#             --trial_name "$TRIAL"

#         echo -e "\nFinalizado: n=$n | LogScale=$log_scale\n"
        
#     done
# done

# echo "Matriz de experimentos concluída com sucesso!"

# # Definição das seeds (você pode alterar os números abaixo)
# SEEDS=(0 31)

# for seed in "${SEEDS[@]}"; do
    
#     TRIAL="PointBERT_ShapeNet_seed_${seed}_False"
    
#     echo "======================================================="
#     echo "EXECUTANDO: Experimento com SEED = $seed"
#     echo "Trial Name: $TRIAL"
#     echo "======================================================="

#     # Execução do torchrun com a flag de seed
#     # Certifique-se de que seu main.py aceita o parâmetro --seed ou seed=
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py \
#         --config configs/Pre-Training/mrl_pointbert.yaml \
#         seed=$seed \
#         --trial_name "$TRIAL"

#     echo -e "\nFinalizado experimento para seed=$seed\n"
    
# done

# echo "Todos os 3 experimentos de seed foram concluídos!"

# # Interrompe o script se houver erro
# set -e

# # Definição da escala de imagens
# NUM_IMGS=(1 2 4 8)

# for n in "${NUM_IMGS[@]}"; do
    
#     TRIAL="PointBERT_ShapeNet_n${n}_5dim"
    
#     echo "======================================================="
#     echo "EXECUTANDO: Experimento com $n Imagens"
#     echo "Trial Name: $TRIAL"
#     echo "======================================================="

#     # Execução direta no terminal
#     CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py \
#         --config configs/Pre-Training/mrl_pointbert.yaml \
#         dataset.num_imgs=$n \
#         --trial_name "$TRIAL"

#     echo -e "\nFinalizado experimento para n=$n\n"
    
# done

# echo "Todos os 4 experimentos foram concluídos!"


# # Interrompe o script se houver erro crítico no shell
# set -e

# # Definições dos parâmetros
# NUM_IMGS=(1 2 4 8)
# MRL_VALS=("True" "False")

# # Cria uma pasta para os logs se não existir
# mkdir -p logs_treinamento

# for n in "${NUM_IMGS[@]}"; do
#     for img_proj in "${MRL_VALS[@]}"; do
#         for text_proj in "${MRL_VALS[@]}"; do
            
#             TRIAL="PointBERT_ShapeNet_logscale_n${n}_I_${img_proj}_T_${text_proj}"
#             LOG_FILE="logs_treinamento/${TRIAL}.log"
            
#             echo "======================================================="
#             echo "EXECUTANDO: Imagens=$n | Img_Proj=$img_proj | Text_Proj=$text_proj"
#             echo "Logs salvos em: $LOG_FILE"
#             echo "======================================================="

#             # O comando 'tee' mostra no terminal E salva no arquivo ao mesmo tempo
#             CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py \
#                 --config configs/Pre-Training/mrl_pointbert.yaml \
#                 dataset.num_imgs=$n \
#                 training.use_mrl_image_proj=$img_proj \
#                 training.use_mrl_text_proj=$text_proj \
#                 --trial_name "$TRIAL" 2>&1 | tee "$LOG_FILE"

#             echo -e "\nFinalizado experimento: $TRIAL\n"
            
#         done
#     done
# done

# echo "Matriz de 16 experimentos completa!"



# # Interrompe o script se houver erro
# set -e

# # Definições dos parâmetros
# NUM_IMGS=(1 2 4 8)
# MRL_VALS=("True" "False")

# for n in "${NUM_IMGS[@]}"; do
#     for img_proj in "${MRL_VALS[@]}"; do
#         for text_proj in "${MRL_VALS[@]}"; do
            
#             # Criando um nome único para o trial para não sobrescrever dados
#             TRIAL="PointBERT_ShapeNet_n${n}_I_${img_proj}_T_${text_proj}"
            
#             echo "======================================================="
#             echo "EXECUTANDO: Imagens=$n | Img_Proj=$img_proj | Text_Proj=$text_proj"
#             echo "Trial Name: $TRIAL"
#             echo "======================================================="

#             CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 main.py \
#                 --config configs/Pre-Training/mrl_pointbert.yaml \
#                 dataset.num_imgs=$n \
#                 training.use_mrl_image_proj=$img_proj \
#                 training.use_mrl_text_proj=$text_proj \
#                 --trial_name "$TRIAL"

#         done
#     done
# done

# echo "Matriz de experimentos completa!"