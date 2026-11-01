import logging
import json
import os
import sys
import h5py
import torch
import numpy as np
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from collections import OrderedDict

import models

# NOTE: make_objaverse_lvis builds a torch.utils.data.distributed.DistributedSampler,
# so this script still needs a process group initialized before running
# (e.g. `torchrun --nproc_per_node=1 retrieval/extract_embeddings_openshape.py ...`).
from data import make_objaverse_lvis

from param import parse_args
from omegaconf import OmegaConf

import data


def clean_ddp_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    return new_state_dict


def load_retrieval_model_openshape(config, checkpoint_path, device):
    logging.info("Loading model...")

    model = models.make(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(clean_ddp_state_dict(checkpoint["state_dict"]))

    model.eval()

    return model


def extract_embeddings(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Uses the function that loads only the backbone, without the MRL heads
    model = load_retrieval_model_openshape(
        config, config.pretrained_model.path, device
    )

    dataloader = make_objaverse_lvis(config)

    # Single list to hold all embeddings of the fixed dimension
    all_embeddings = []
    shape_model_to_idx = {}
    current_idx = 0

    logging.info("Extracting embeddings...")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_ids = batch["name"]

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Feature extraction via the backbone
            if not config.model.get("use_dense", False):
                pred_feat = model(
                    batch["xyz"],
                    batch["features"],
                    device=device,
                    quantization_size=config.model.voxel_size,
                )
            else:
                pred_feat = model(batch["xyz_dense"], batch["features_dense"])

            # L2 normalization directly on the output embedding
            pred_feat = F.normalize(pred_feat, dim=1)

            emb_np = pred_feat.cpu().numpy().astype(np.float32)
            all_embeddings.append(emb_np)

            # ID mapping
            for i, model_id in enumerate(batch_ids):
                shape_model_to_idx[model_id] = current_idx + i

            current_idx += len(batch_ids)

    logging.info("Saving H5...")

    h5_path = config.embeddings.openshape_h5
    json_path = config.embeddings.openshape_json
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)

    with h5py.File(h5_path, "w") as h5f:
        # Concatenates the list into a single (N, embedding_dim) matrix
        matrix = np.concatenate(all_embeddings, axis=0)

        print(f"shape_feat: {matrix.shape}")

        # Saves to a single fixed dataset, without the MRL dimension suffix
        h5f.create_dataset("shape_feat", data=matrix)

    with open(json_path, "w") as f:
        json.dump(shape_model_to_idx, f)

    logging.info("Extraction finished!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cli_args, extras = parse_args(sys.argv[1:])
    config = OmegaConf.load(cli_args.config)

    extract_embeddings(config)
