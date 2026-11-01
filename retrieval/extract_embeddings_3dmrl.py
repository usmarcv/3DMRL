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
from trainers.MRL import MRL_Projection_Layer

# NOTE: make_objaverse_lvis builds a torch.utils.data.distributed.DistributedSampler,
# so this script still needs a process group initialized before running
# (e.g. `torchrun --nproc_per_node=1 retrieval/extract_embeddings_3dmrl.py ...`).
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


def load_retrieval_model(
    config,
    checkpoint_path,
    device
):

    logging.info("Loading model...")

    model = models.make(config).to(device)

    mrl_heads = MRL_Projection_Layer(
        nesting_list=config.mrl.nesting_dims,
        out_dim=config.mrl.out_dim,
        efficient=config.mrl.efficient
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        clean_ddp_state_dict(
            checkpoint["state_dict"]
        )
    )

    mrl_heads.load_state_dict(
        clean_ddp_state_dict(
            checkpoint["mrl_heads"]
        )
    )

    model.eval()
    mrl_heads.eval()

    return model, mrl_heads


def load_retrieval_model_openshape(
    config,
    checkpoint_path,
    device
):

    logging.info("Loading model...")

    model = models.make(config).to(device)


    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        clean_ddp_state_dict(
            checkpoint["state_dict"]
        )
    )



    model.eval()


    return model



def extract_embeddings(config):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, mrl_heads = load_retrieval_model(
        config,
        config.pretrained_model.path,
        device
    )

    # train_loader = data.make(config, 'train', 0, 1)

    dataloader = make_objaverse_lvis(config)

    nesting_dims = config.mrl.nesting_dims

    all_embeddings = {
        dim: [] for dim in nesting_dims
    }

    shape_model_to_idx = {}

    current_idx = 0

    logging.info(
        "Extracting embeddings..."
    )

    with torch.no_grad():

        for batch in tqdm(dataloader):

            batch_ids = batch["name"]

            for k, v in batch.items():

                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Backbone
            if not config.model.get(
                "use_dense",
                False
            ):

                pred_feat = model(
                    batch["xyz"],
                    batch["features"],
                    device=device,
                    quantization_size=config.model.voxel_size
                )

            else:

                pred_feat = model(
                    batch["xyz_dense"],
                    batch["features_dense"]
                )

            # List:
            # [
            #   (B,1280),
            #   (B,1280),
            #   ...
            # ]
            mrl_embeddings = mrl_heads(
                pred_feat
            )

            for dim, emb in zip(
                nesting_dims,
                mrl_embeddings
            ):

                print(
                    f"Dim {dim} -> {emb.shape}"
                )

                # ALWAYS 1280
                assert (
                    emb.shape[-1]
                    == config.mrl.out_dim
                )

                emb = F.normalize(
                    emb,
                    dim=1
                )

                emb_np = (
                    emb
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

                all_embeddings[dim].append(
                    emb_np
                )

            # Mapping IDs
            for i, model_id in enumerate(
                batch_ids
            ):

                shape_model_to_idx[
                    model_id
                ] = current_idx + i

            current_idx += len(batch_ids)

    logging.info("Saving H5...")

    h5_path = config.embeddings.mrl_h5
    json_path = config.embeddings.mrl_json
    os.makedirs(os.path.dirname(os.path.abspath(h5_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)

    with h5py.File(
        h5_path,
        "w"
    ) as h5f:

        for dim in nesting_dims:

            matrix = np.concatenate(
                all_embeddings[dim],
                axis=0
            )

            print(
                f"shape_feat_{dim}: "
                f"{matrix.shape}"
            )

            h5f.create_dataset(
                f"shape_feat_{dim}",
                data=matrix
            )

    with open(
        json_path,
        "w"
    ) as f:

        json.dump(
            shape_model_to_idx,
            f
        )

    logging.info(
        "Extraction finished!"
    )


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    cli_args, extras = parse_args(
        sys.argv[1:]
    )

    config = OmegaConf.load(
        cli_args.config
    )

    extract_embeddings(config)
