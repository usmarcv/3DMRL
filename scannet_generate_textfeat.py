import torch
import open_clip
import numpy as np
import os

# 1. Lista de classes na MESMA ordem do seu data.py
categories = ['sink', 'chair', 'toilet', 'door', 'desk', 'shower curtain', 'sofa', 'window',
                           'table', 'curtain', 'picture', 'cabinet', 'refrigerator', 'bookshelf', 'bed',
                           'bathtub', 'counter']

print("Carregando o modelo OpenCLIP (ViT-bigG-14)...")
# CORREÇÃO: Ignoramos as transforms de imagem e pegamos apenas o modelo
model, _, _ = open_clip.create_model_and_transforms(
    'ViT-bigG-14', pretrained='laion2b_s39b_b160k'
)
model = model.cuda().eval()

# CORREÇÃO: Buscando o tokenizer de texto correto para o modelo
tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

# 3. Engenharia de Prompts
templates = [
    'a photo of a {}',
    'a 3d model of a {}',
    'a point cloud of a {}',
    'the shape of a {}',
]

clip_cat_feat = []

print("Extraindo embeddings de texto para as 20 classes...")
with torch.no_grad():
    for category in categories:
        cat_name = category.replace('_', ' ')
        
        # Gera os textos baseados nos templates
        texts = [template.format(cat_name) for template in templates]
        tokens = tokenizer(texts).cuda()
        
        # Passa pelo codificador de texto do CLIP e normaliza
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Tira a média dos embeddings dos templates
        mean_feature = text_features.mean(dim=0)
        mean_feature /= mean_feature.norm(dim=-1, keepdim=True)
        
        clip_cat_feat.append(mean_feature.cpu().numpy())

# 4. Salva a matriz final (20, 1280)
clip_cat_feat = np.array(clip_cat_feat)
print("Formato final da matriz gerada:", clip_cat_feat.shape)

# Garante que a pasta de destino existe antes de salvar
os.makedirs('data/meta_data', exist_ok=True)
np.save('data/meta_data/scannet_cat_name_pt_feat.npy', clip_cat_feat)
print("Arquivo 'data/meta_data/scannet_cat_name_pt_feat.npy' salvo com sucesso!")

# import torch
# import open_clip
# import numpy as np

# # 1. Lista de classes na MESMA ordem do seu data.py
# categories = ['sink', 'chair', 'toilet', 'door', 'desk', 'shower curtain', 'sofa', 'window',
#                            'table', 'curtain', 'picture', 'cabinet', 'refrigerator', 'bookshelf', 'bed',
#                            'bathtub', 'counter']

# # 2. Carrega o modelo OpenCLIP correto para extrair características de 1280 dimensões
# print("Carregando o modelo OpenCLIP (ViT-bigG-14)...")
# model, _, tokenizer = open_clip.create_model_and_transforms(
#     'ViT-bigG-14', pretrained='laion2b_s39b_b160k'
# )
# model = model.cuda().eval()

# # 3. Engenharia de Prompts (padrão usado no OpenShape/CLIP para melhorar acurácia)
# templates = [
#     'a photo of a {}',
#     'a 3d model of a {}',
#     'a point cloud of a {}',
#     'the shape of a {}',
# ]

# clip_cat_feat = []

# with torch.no_grad():
#     for category in categories:
#         # Corrige nomes compostos se necessário para o texto fluir melhor
#         cat_name = category.replace('_', ' ')
        
#         # Gera os textos baseados nos templates
#         texts = [template.format(cat_name) for template in templates]
#         tokens = tokenizer(texts).cuda()
        
#         # Passa pelo codificador de texto do CLIP e normaliza
#         text_features = model.encode_text(tokens)
#         text_features /= text_features.norm(dim=-1, keepdim=True)
        
#         # Tira a média dos embeddings dos templates para criar um vetor robusto
#         mean_feature = text_features.mean(dim=0)
#         mean_feature /= mean_feature.norm(dim=-1, keepdim=True)
        
#         clip_cat_feat.append(mean_feature.cpu().numpy())

# # 4. Salva a matriz final que terá o formato exato (20, 1280)
# clip_cat_feat = np.array(clip_cat_feat)
# print("Formato final da matriz gerada:", clip_cat_feat.shape) # Deve ser (20, 1280)

# np.save('data/meta_data/scannet_cat_name_pt_feat.npy', clip_cat_feat)
# print("Arquivo de texto do ScanNet salvo com sucesso!")


# # import torch
# # import open_clip
# # import numpy as np
# # import os

# # # =====================================================================
# # # 1. SUAS CATEGORIAS EXATAS (K categories)
# # # =====================================================================
# # categories = [
# #     'sink', 'chair', 'toilet', 'door', 'desk', 'shower curtain', 'sofa', 'window',
# #     'table', 'curtain', 'picture', 'cabinet', 'refrigerator', 'bookshelf', 'bed',
# #     'bathtub', 'counter'
# # ]

# # # =====================================================================
# # # 2. EXTRAÇÃO DAS EMBEDDINGS VIA OPENCLIP
# # # =====================================================================
# # print("Carregando o modelo OpenCLIP (ViT-bigG-14)...")
# # model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
# # model = model.cuda().eval()
# # tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

# # clip_cat_feat = []

# # print("Iniciando extração usando o prompt exato 'point cloud of {CLASS}'...")
# # with torch.no_grad():
# #     for cat in categories:
# #         # Cria o prompt único conforme o artigo
# #         prompt = f"point cloud of {cat}"
# #         print(f"-> Classe [{cat:<15}]: Prompt gerado -> '{prompt}'")
        
# #         # Tokenização e inferência no CLIP
# #         tokens = tokenizer([prompt]).cuda()
# #         text_feature = model.encode_text(tokens) # Shape: (1, 1280)
        
# #         # Normalização L2 no vetor gerado
# #         text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        
# #         # Remove a dimensão extra (1, 1280) -> (1280,) e envia para a CPU
# #         clip_cat_feat.append(text_feature.squeeze().cpu().numpy())

# # clip_cat_feat = np.array(clip_cat_feat)
# # print("\nConcluído!")
# # print("Formato final da matriz de texto .npy:", clip_cat_feat.shape) # Exatamente (17, 1280)

# # # =====================================================================
# # # 3. SALVAMENTO
# # # =====================================================================
# # output_dir = 'data/meta_data'
# # os.makedirs(output_dir, exist_ok=True)

# # # Mudei levemente o nome do arquivo para refletir o novo prompt
# # output_path = os.path.join(output_dir, 'scannet_pointcloud_prompts.npy')

# # np.save(output_path, clip_cat_feat)
# # print(f"Arquivo salvo com sucesso em: {output_path}")


# # # import torch
# # # import open_clip
# # # import numpy as np
# # # import os

# # # # =====================================================================
# # # # 1. SEUS TEMPLATES EXATOS (MODELNET40 / SHAPENET)
# # # # =====================================================================
# # # user_templates = [
# # #     "a point cloud model of {}.",
# # #     "There is a {} in the scene.",
# # #     "There is the {} in the scene.",
# # #     "a photo of a {} in the scene.",
# # #     "a photo of the {} in the scene.",
# # #     "a photo of one {} in the scene.",
# # #     "itap of a {}.",
# # #     "itap of my {}.",
# # #     "itap of the {}.",
# # #     "a photo of a {}.",
# # #     "a photo of my {}.",
# # #     "a photo of the {}.",
# # #     "a photo of one {}.",
# # #     "a photo of many {}.",
# # #     "a good photo of a {}.",
# # #     "a good photo of the {}.",
# # #     "a bad photo of a {}.",
# # #     "a bad photo of the {}.",
# # #     "a photo of a nice {}.",
# # #     "a photo of the nice {}.",
# # #     "a photo of a cool {}.",
# # #     "a photo of the cool {}.",
# # #     "a photo of a weird {}.",
# # #     "a photo of the weird {}.",
# # #     "a photo of a small {}.",
# # #     "a photo of the small {}.",
# # #     "a photo of a large {}.",
# # #     "a photo of the large {}.",
# # #     "a photo of a clean {}.",
# # #     "a photo of the clean {}.",
# # #     "a photo of a dirty {}.",
# # #     "a photo of the dirty {}.",
# # #     "a bright photo of a {}.",
# # #     "a bright photo of the {}.",
# # #     "a dark photo of a {}.",
# # #     "a dark photo of the {}.",
# # #     "a photo of a hard to see {}.",
# # #     "a photo of the hard to see {}.",
# # #     "a low resolution photo of a {}.",
# # #     "a low resolution photo of the {}.",
# # #     "a cropped photo of a {}.",
# # #     "a cropped photo of the {}.",
# # #     "a close-up photo of a {}.",
# # #     "a close-up photo of the {}.",
# # #     "a jpeg corrupted photo of a {}.",
# # #     "a jpeg corrupted photo of the {}.",
# # #     "a blurry photo of a {}.",
# # #     "a blurry photo of the {}.",
# # #     "a pixelated photo of a {}.",
# # #     "a pixelated photo of the {}.",
# # #     "a black and white photo of the {}.",
# # #     "a black and white photo of a {}.",
# # #     "a plastic {}.",
# # #     "the plastic {}.",
# # #     "a toy {}.",
# # #     "the toy {}.",
# # #     "a plushie {}.",
# # #     "the plushie {}.",
# # #     "a cartoon {}.",
# # #     "the cartoon {}.",
# # #     "an embroidered {}.",
# # #     "the embroidered {}.",
# # #     "a painting of the {}.",
# # #     "a painting of a {}."
# # # ]

# # # def generate_extensive_prompts(class_name, synonyms):
# # #     all_prompts = []
# # #     # Cria uma lista única com o nome principal e todos os sinônimos da classe
# # #     all_names = list(set([class_name] + synonyms))
    
# # #     for name in all_names:
# # #         # Aplica o nome/sinônimo diretamente em cada um dos seus templates
# # #         for template in user_templates:
# # #             all_prompts.append(template.format(name))
                
# # #     return list(set(all_prompts)) # Remove duplicatas estruturais

# # # # =====================================================================
# # # # 2. SEUS SINÔNIMOS DO LVIS / SCANNET
# # # # =====================================================================
# # # lvis_synonyms = {
# # #     'sink': ['sink', 'washbasin', 'basin', 'handbasin', 'bathroom sink', 'kitchen sink'],
# # #     'chair': ['chair', 'armchair', 'seat', 'stool', 'office chair', 'folding chair'],
# # #     'toilet': ['toilet', 'commode', 'water closet', 'toilet bowl'],
# # #     'door': ['door', 'doorway', 'gate', 'sliding door'],
# # #     'desk': ['desk', 'writing desk', 'office desk', 'worktable', 'study table'],
# # #     'shower curtain': ['shower curtain', 'bathroom curtain', 'shower drape'],
# # #     'sofa': ['sofa', 'couch', 'settee', 'lounge', 'davenport', 'love seat'],
# # #     'window': ['window', 'windowpane', 'casement', 'glass window'],
# # #     'table': ['table', 'dining table', 'coffee table', 'side table', 'kitchen table'],
# # #     'curtain': ['curtain', 'drape', 'drapery', 'window curtain', 'blind'],
# # #     'picture': ['picture', 'painting', 'photo frame', 'poster', 'artwork', 'wall art'],
# # #     'cabinet': ['cabinet', 'cupboard', 'closet', 'wardrobe', 'filing cabinet', 'locker'],
# # #     'refrigerator': ['refrigerator', 'fridge', 'freezer', 'wine fridge'],
# # #     'bookshelf': ['bookshelf', 'bookcase', 'bookstands', 'shelving unit'],
# # #     'bed': ['bed', 'bedstead', 'mattress', 'bunk bed', 'double bed'],
# # #     'bathtub': ['bathtub', 'tub', 'bath', 'soaking tub'],
# # #     'counter': ['counter', 'countertop', 'bar counter', 'kitchen counter']
# # # }

# # # # Mantém a ordem exata das 17 classes de avaliação
# # # categories = list(lvis_synonyms.keys())

# # # # =====================================================================
# # # # 3. EXTRAÇÃO DAS EMBEDDINGS VIA OPENCLIP
# # # # =====================================================================
# # # print("Carregando o modelo OpenCLIP (ViT-bigG-14)...")
# # # model, _, _ = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
# # # model = model.cuda().eval()
# # # tokenizer = open_clip.get_tokenizer('ViT-bigG-14')

# # # clip_cat_feat = []

# # # print("Iniciando extração baseada estritamente nos seus templates...")
# # # with torch.no_grad():
# # #     for cat in categories:
# # #         # Gera os prompts usando apenas os seus templates
# # #         prompts = generate_extensive_prompts(cat, lvis_synonyms[cat])
# # #         print(f"-> Classe [{cat:<15}]: Gerados {len(prompts)} prompts com seus templates.")
        
# # #         # Tokenização e inferência no CLIP
# # #         tokens = tokenizer(prompts).cuda()
# # #         text_features = model.encode_text(tokens) # Shape: (Num_Prompts, 1280)
        
# # #         # Normalização individual por prompt
# # #         text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
# # #         # Média geométrica do conjunto de vetores da classe
# # #         mean_feature = text_features.mean(dim=0)
        
# # #         # Renormalização final do vetor resultante
# # #         mean_feature = mean_feature / mean_feature.norm(dim=-1, keepdim=True)
        
# # #         clip_cat_feat.append(mean_feature.cpu().numpy())

# # # clip_cat_feat = np.array(clip_cat_feat)
# # # print("\nConcluído!")
# # # print("Formato final da matriz de texto .npy:", clip_cat_feat.shape) # Exatamente (17, 1280)

# # # # Salvamento
# # # output_dir = 'data/meta_data'
# # # os.makedirs(output_dir, exist_ok=True)
# # # output_path = os.path.join(output_dir, 'scannet_pure_user_prompts.npy')

# # # np.save(output_path, clip_cat_feat)
# # # print(f"Arquivo salvo com sucesso em: {output_path}")