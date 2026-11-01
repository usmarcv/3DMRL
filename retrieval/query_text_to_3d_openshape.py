import sys
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import h5py

from param import parse_args
from omegaconf import OmegaConf

from retrieval.faiss_utils import build_index, load_id_mapping, load_openclip_text_encoder


def setup_faiss_databases(h5_path, json_path):
    logging.info("Loading FAISS database...")

    with h5py.File(h5_path, "r") as h5f:
        # Loads directly the single-dimension dataset produced by OpenShape
        dataset_name = "shape_feat"

        if dataset_name not in h5f:
            raise KeyError(f"Dataset '{dataset_name}' not found in the H5 file.")

        shape_embeddings = h5f[dataset_name][:]
        print(f"Loaded {dataset_name} with shape: {shape_embeddings.shape}")

        assert shape_embeddings.ndim == 2
        out_dim = shape_embeddings.shape[1]  # Dynamically detected embedding dimension

        # Creates the permanent index based on the detected native dimension
        index, _ = build_index(shape_embeddings)

        logging.info(f"Embeddings indexed successfully! Total: {index.ntotal} (Dim: {out_dim})")

    shape_model_to_idx, idx_to_shape_id = load_id_mapping(json_path)

    return index, idx_to_shape_id, out_dim


def run_interactive_retrieval(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Loads the text model
    text_model, tokenizer = load_openclip_text_encoder(device)

    # Initializes the database using the OpenShape embedding files
    index_faiss, idx_to_shape_id, out_dim = setup_faiss_databases(
        h5_path=config.embeddings.openshape_h5,
        json_path=config.embeddings.openshape_json
    )

    print("\n" + "=" * 60)
    print(f"RETRIEVAL TEXT-INPUT TO 3D SHAPE (EMBEDDING DIMENSION: {out_dim})")
    print("=" * 60)

    while True:
        text_input = input("\nEnter a description to search models by (or 'exit' to quit): ")

        if text_input.lower() in ["exit"]:
            break

        if not text_input.strip():
            continue

        text_tokens = tokenizer([text_input]).to(device)

        with torch.no_grad():
            # Extracts the OpenCLIP embedding (shape: [1, 1280])
            query_emb = text_model.encode_text(text_tokens).float()

            # Normalizes the query vector in PyTorch
            query_emb = F.normalize(query_emb, dim=-1)

        query_emb = query_emb.cpu().numpy().astype(np.float32)
        query_emb = np.ascontiguousarray(query_emb)

        print(f"\nResults for: '{text_input}'")
        print(f"Query shape: {query_emb.shape}")

        K = 5
        # Searches directly in the FAISS index
        distances, indices = index_faiss.search(query_emb, K)

        print(f"\n--- Top {K} Results ---")
        for rank in range(K):
            faiss_idx = indices[0][rank]
            score = distances[0][rank]
            shape_id = idx_to_shape_id[faiss_idx]

            print(
                f"[{rank+1}] "
                f"Score: {score:.4f} "
                f"| ID: {shape_id}"
            )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s"
    )

    cli_args, extras = parse_args(sys.argv[1:])
    config = OmegaConf.load(cli_args.config)

    run_interactive_retrieval(config)
