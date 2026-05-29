import os
import json
import numpy as np
import torch
from tqdm import tqdm

# =====================================================================
# CONFIGURAÇÃO DE CAMINHOS E CLASSES
# =====================================================================
CATEGORIES = [
    'sink', 'chair', 'toilet', 'door', 'desk', 'shower curtain', 'sofa', 'window',
    'table', 'curtain', 'picture', 'cabinet', 'refrigerator', 'bookshelf', 'bed',
    'bathtub', 'counter'
]

# Configure os caminhos de acordo com a imagem do seu terminal:
RAW_DATA_DIR = "data/scannetv2/val"              # Diretório que contém os arquivos .pth das cenas
OUTPUT_DIR = "data/meta_data/scannet_processed"   # Onde os arquivos .npy finais serão guardados
OUTPUT_JSON = "data/meta_data/scannet_openshape_split.json" # O arquivo de mapeamento do DataLoader

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_single_object(raw_xyz, raw_rgb):
    """
    Aplica o pipeline exato do OpenShape: amostragem de 10k pontos,
    centralização e normalização para [-1, 1] e [0, 1].
    """
    num_target_points = 10000
    n = raw_xyz.shape[0]
    
    if n < 100: # Ignora objetos ruidosos ou com pouquíssimos pontos
        return None

    # 1. Amostragem fixa (Subsampling / Padding se necessário)
    if n >= num_target_points:
        idx = np.random.choice(n, num_target_points, replace=False)
    else:
        idx = np.random.choice(n, num_target_points, replace=True)
        
    xyz = raw_xyz[idx]
    rgb = raw_rgb[idx]

    # 2. Centralização na Origem (0, 0, 0)
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid

    # 3. Normalização rígida para o intervalo [-1, 1]
    max_distance = np.max(np.sqrt(np.sum(xyz ** 2, axis=-1)))
    if max_distance > 0:
        xyz = xyz / max_distance

    # 4. Normalização de Cores para [0, 1]
    if np.max(rgb) > 1.0:
        rgb = rgb / 255.0

    return {
        "xyz": xyz.astype(np.float32),
        "rgb": rgb.astype(np.float32)
    }

# =====================================================================
# LOOP PRINCIPAL DE PROCESSAMENTO
# =====================================================================
def run_preprocessing():
    new_split_annotations = []
    
    # Procura por todos os ficheiros .pth diretamente na diretoria fornecida
    pth_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.pth')]
    print(f"Encontradas {len(pth_files)} cenas .pth para processamento.")

    for file_name in tqdm(pth_files, desc="Processando Cenas ScanNet"):
        scene_id = os.path.splitext(file_name)[0]
        file_path = os.path.join(RAW_DATA_DIR, file_name)
        
        try:
            # Carrega o dicionário da cena via PyTorch de forma muito veloz
            scene_data = torch.load(file_path, map_location='cpu')
        except Exception as e:
            print(f"Erro ao carregar o ficheiro {file_name}: {e}")
            continue

        # Dependendo de como o seu arquivo .pth foi salvo, as chaves podem mudar.
        # Geralmente os padrões do ScanNet salvam como:
        # scene_data['mesh_vertices'] ou scene_data['coord'] e scene_data['color']
        # Também precisamos mapear o array de instâncias/labels.
        
        # Ajuste adaptativo de chaves comuns do ScanNet .pth:
        all_xyz = scene_data.get('coord', scene_data.get('mesh_vertices', None))
        all_rgb = scene_data.get('color', scene_data.get('mesh_colors', None))
        all_labels = scene_data.get('semantic_labels', scene_data.get('labels', None))
        instance_ids = scene_data.get('instance_labels', scene_data.get('instance_ids', None))

        # Se os dados estiverem guardados num formato estruturado diferente (ex: lista de objetos),
        # pode ser necessário ajustar estas chaves para bater com o seu dicionário de entrada.
        if all_xyz is None or all_labels is None:
            # Caso os seus tensores estejam em formato NumPy dentro do .pth:
            continue
            
        # Converte para instâncias NumPy caso sejam tensores PyTorch
        if isinstance(all_xyz, torch.Tensor): all_xyz = all_xyz.numpy()
        if isinstance(all_rgb, torch.Tensor): all_rgb = all_rgb.numpy()
        if isinstance(all_labels, torch.Tensor): all_labels = all_labels.numpy()
        if isinstance(instance_ids, torch.Tensor): instance_ids = instance_ids.numpy()

        # Descobre os IDs únicos de instâncias presentes nesta cena
        unique_instances = np.unique(instance_ids)

        for inst_id in unique_instances:
            if inst_id < 0: # Ignora fundo/ruído sem instância definida
                continue
                
            # Cria a máscara para isolar os pontos pertencentes a esta instância
            inst_mask = (instance_ids == inst_id)
            
            # Descobre a classe semântica correspondente a este objeto
            # Pegamos o primeiro label válido encontrado na máscara
            obj_label_idx = all_labels[inst_mask][0]
            
            # Mapeamento do índice numérico para a string da classe (ex: 3 -> 'chair')
            # NOTA: Certifique-se de que a sua lista interna de mapeamento bate com o dataset bruto.
            # Se a classe já vier mapeada como String, pode usar diretamente.
            # Supondo que você tem o mapeamento de IDs numéricos para os nomes reais:
            if hasattr(self, 'label_map'): 
                label_string = label_map[obj_label_idx].strip().lower()
            else:
                # Caso o seu array semântico já possua strings ou precise de tratamento:
                label_string = str(obj_label_idx).strip().lower()

            # FILTRO CRÍTICO: Processa apenas se o objeto estiver contido nas 17 classes
            if label_string not in CATEGORIES:
                continue

            obj_xyz = all_xyz[inst_mask]
            obj_rgb = all_rgb[inst_mask]

            # Executa o pipeline geométrico do OpenShape
            processed_data = process_single_object(obj_xyz, obj_rgb)

            if processed_data is not None:
                out_file_name = f"{scene_id}_inst_{inst_id}_{label_string}.npy"
                save_path = os.path.join(OUTPUT_DIR, out_file_name)

                # Salva o resultado final pronto para ser consumido
                np.save(save_path, processed_data)

                # Grava no índice JSON para o dataset loader
                new_split_annotations.append({
                    "file_name": out_file_name,
                    "category": label_string
                })

    # Exporta o JSON final limpo
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(new_split_annotations, f, indent=4)

    print(f"\n[SUCESSO] Conversão finalizada!")
    print(f"Ficheiros gravados em .npy: {len(new_split_annotations)}")
    print(f"Metadados estruturados em: {OUTPUT_JSON}")

if __name__ == "__main__":
    run_preprocessing()