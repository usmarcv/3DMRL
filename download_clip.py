import torch
import open_clip
import os

# Caminho onde vamos salvar
save_path = "ViT-bigG-14-laion2b_s39b_b160k.pt"

# ============================
# 1. Baixar o modelo do OpenCLIP
# ============================
print("🔽 Baixando modelo do OpenCLIP...")
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-bigG-14',
    pretrained='laion2b_s39b_b160k',
    cache_dir="data/pre-trained-clips/open-clip"
)

# ============================
# 2. Salvar os pesos em .pt
# ============================
print(f"💾 Salvando pesos em {save_path} ...")
torch.save(model.state_dict(), save_path)


# ============================
# 3. Recarregar só do arquivo local
# ============================
print("♻️ Recarregando do .pt local...")
# cria a arquitetura "vazia"
model_local, _, preprocess_local = open_clip.create_model_and_transforms(
    'ViT-bigG-14',
    pretrained=None  # não baixa nada
)

# carrega os pesos
state_dict = torch.load(save_path, map_location="cpu")
model_local.load_state_dict(state_dict)
model_local.eval()

print("✅ Modelo recarregado com sucesso do arquivo local!")

