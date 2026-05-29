import sys
import json
import h5py
import faiss
import torch
import open_clip
import logging
import numpy as np
import torch.nn.functional as F

from param import parse_args
from omegaconf import OmegaConf


def load_text_pipeline(config, device):
    logging.info("Loading OpenCLIP...")
    
    # ViT-bigG-14 possui dimensionalidade nativa de 1280
    text_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14",
        pretrained="laion2b_s39b_b160k",
        precision="fp16",
    )

    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
    text_model = text_model.to(device).eval()
    
    return text_model, tokenizer


def setup_faiss_databases(h5_path, json_path):
    logging.info("Loading FAISS database...")

    with h5py.File(h5_path, "r") as h5f:
        # Carrega diretamente o dataset de dimensão única gerado pelo OpenShape
        dataset_name = "shape_feat"
        
        if dataset_name not in h5f:
            raise KeyError(f"Dataset '{dataset_name}' não encontrado no arquivo H5.")
            
        shape_embeddings = h5f[dataset_name][:]
        print(f"Loaded {dataset_name} with shape: {shape_embeddings.shape}")

        assert shape_embeddings.ndim == 2
        out_dim = shape_embeddings.shape[1]  # Obtém dinamicamente a dimensão do embedding

        shape_embeddings = np.ascontiguousarray(
            shape_embeddings.astype(np.float32)
        )

        faiss.normalize_L2(shape_embeddings)

        # Cria o índice permanente baseado na dimensão nativa detectada
        index = faiss.IndexFlatIP(out_dim)
        index.add(shape_embeddings)

        logging.info(f"Embeddings indexados com sucesso! Total: {index.ntotal} (Dim: {out_dim})")

    with open(json_path, "r") as f:
        shape_model_to_idx = json.load(f)

    idx_to_shape_id = {v: k for k, v in shape_model_to_idx.items()}

    return index, idx_to_shape_id, out_dim


def run_interactive_retrieval(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Carrega o modelo de texto
    text_model, tokenizer = load_text_pipeline(config, device)

    # Inicializa o banco de dados usando os arquivos atualizados do OpenShape
    index_faiss, idx_to_shape_id, out_dim = setup_faiss_databases(
        h5_path="shape_embeddings_openshape.h5",
        json_path="shape_model_to_idx_openshape.json"
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
            # Extrai o embedding do OpenCLIP (shape: [1, 1280])
            query_emb = text_model.encode_text(text_tokens).float()
            
            # Normaliza o vetor de query no PyTorch
            query_emb = F.normalize(query_emb, dim=-1)

        query_emb = query_emb.cpu().numpy().astype(np.float32)
        query_emb = np.ascontiguousarray(query_emb)

        print(f"\nResultados para: '{text_input}'")
        print(f"Query shape: {query_emb.shape}")

        K = 5
        # Realiza a busca diretamente no índice FAISS
        distances, indices = index_faiss.search(query_emb, K)

        print(f"\n--- Top {K} Resultados ---")
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