import logging
import json
import h5py
import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from collections import OrderedDict

import models
from data_retrieval import make_objaverse_lvis

from param import parse_args
from omegaconf import OmegaConf
import sys

import data


def clean_ddp_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    return new_state_dict


def load_retrieval_model_openshape(config, checkpoint_path, device):
    logging.info("Carregando modelo...")

    model = models.make(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(clean_ddp_state_dict(checkpoint["state_dict"]))

    model.eval()

    return model


def extract_embeddings(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Usando a função que carrega apenas o backbone, sem as cabeças MRL
    model = load_retrieval_model_openshape(
        config, config.pretrained_model.path, device
    )

    dataloader = make_objaverse_lvis(config)

    # Lista única para armazenar todos os embeddings da dimensão padrão
    all_embeddings = []
    shape_model_to_idx = {}
    current_idx = 0

    logging.info("Extraindo embeddings...")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_ids = batch["name"]

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Extração das features pelo Backbone
            if not config.model.get("use_dense", False):
                pred_feat = model(
                    batch["xyz"],
                    batch["features"],
                    device=device,
                    quantization_size=config.model.voxel_size,
                )
            else:
                pred_feat = model(batch["xyz_dense"], batch["features_dense"])

            # Normalização L2 diretamente no embedding de saída
            pred_feat = F.normalize(pred_feat, dim=1)

            emb_np = pred_feat.cpu().numpy().astype(np.float32)
            all_embeddings.append(emb_np)

            # Mapeamento dos IDs
            for i, model_id in enumerate(batch_ids):
                shape_model_to_idx[model_id] = current_idx + i

            current_idx += len(batch_ids)

    logging.info("Salvando H5...")

    with h5py.File("shape_embeddings_openshape.h5", "w") as h5f:
        # Concatena a lista em uma única matriz (N, dim_embedding)
        matrix = np.concatenate(all_embeddings, axis=0)

        print(f"shape_feat: {matrix.shape}")

        # Salva em um único dataset fixo, sem o sufixo de dimensão do MRL
        h5f.create_dataset("shape_feat", data=matrix)

    with open("shape_model_to_idx_openshape.json", "w") as f:
        json.dump(shape_model_to_idx, f)

    logging.info("Extração finalizada!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cli_args, extras = parse_args(sys.argv[1:])
    config = OmegaConf.load(cli_args.config)

    extract_embeddings(config)