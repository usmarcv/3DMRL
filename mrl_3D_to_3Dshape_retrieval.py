import sys
import json
import h5py
import faiss
import logging
import os
import numpy as np

from param import parse_args
from omegaconf import OmegaConf


def setup_faiss_databases(h5_path, json_path, nesting_dims):
    logging.info("Loading and slicing FAISS databases for 3D-to-3D (Scenario 1)...")

    indices_faiss = {}
    database_embeddings = {}

    with h5py.File(h5_path, "r") as h5f:
        for dim in nesting_dims:
            dataset_name = f"shape_feat_{dim}"
            
            if dataset_name not in h5f:
                logging.warning(f"Dataset {dataset_name} not found in H5 file. Skipping.")
                continue
                
            shape_embeddings = h5f[dataset_name][:, :dim]
            print(f"-> {dataset_name} successfully sliced to shape: {shape_embeddings.shape}")

            shape_embeddings = np.ascontiguousarray(shape_embeddings.astype(np.float32))
            faiss.normalize_L2(shape_embeddings)
            database_embeddings[dim] = shape_embeddings

            index = faiss.IndexFlatIP(dim) # 512
            index.add(shape_embeddings)
            indices_faiss[dim] = index
            logging.info(f"   FAISS index created for Dim {dim}. Total items: {index.ntotal}")

    with open(json_path, "r") as f:
        shape_model_to_idx = json.load(f)

    idx_to_shape_id = {v: k for k, v in shape_model_to_idx.items()}

    return indices_faiss, database_embeddings, idx_to_shape_id, shape_model_to_idx


def run_interactive_3d_retrieval(config, output_filename):
    nesting_dims = config.mrl.nesting_dims

    # Derive the IDs-only filename from the main output filename parameter
    base, ext = os.path.splitext(output_filename)
    ids_only_filename = f"{base}_ids{ext}"

    indices_faiss, database_embeddings, idx_to_shape_id, shape_model_to_idx = setup_faiss_databases(
        h5_path="mrl_shape_embeddings.h5",
        json_path="mrl_shape_model_to_idx.json",
        nesting_dims=nesting_dims
    )

    print("\n" + "=" * 60)
    print("3D-TO-3D SHAPE RETRIEVAL (MATRYOSHKA ENGINES)")
    print("=" * 60)
    print(f"Saving full logs to: {output_filename}")
    print(f"Saving IDs-only list to: {ids_only_filename}")

    while True:
        sample_ids = list(shape_model_to_idx.keys())[:3]
        print(f"\nSample of valid IDs in your database: {sample_ids}")
        
        shape_id_query = input("Enter the 3D object ID for query (or 'exit'): ").strip()

        if shape_id_query.lower() in ["exit", "quit", "sair"]:
            break

        if shape_id_query not in shape_model_to_idx:
            print("Error: This ID does not exist in the JSON file.")
            continue

        query_idx = shape_model_to_idx[shape_id_query]
        
        print("\nChoose search strategy:")
        print("[1] Run Matryoshka Cascade Search (Fast low-dim filter -> High-dim re-rank)")
        print("[2] Compare accuracy of each dimension isolated")
        option = input("Selected option: ").strip()

        K = 5 # Number of neighbors to return

        # Open both files: one for full logs, one exclusively for IDs
        with open(output_filename, "a", encoding="utf-8") as f_log, \
             open(ids_only_filename, "a", encoding="utf-8") as f_ids:
            
            if option == "1":
                dim_coarse = nesting_dims[0]
                dim_fine = nesting_dims[-1]
                top_n_candidates = 50          
                
                header_cascade = (
                    f"\n========================================\n"
                    f"QUERY: {shape_id_query} | STRATEGY: Matryoshka Cascade\n"
                    f"========================================\n"
                    f"[Stage 1] Filtering top {top_n_candidates} using only {dim_coarse} dimensions...\n"
                    f"[Stage 2] Re-ranking candidates using max dimension ({dim_fine})...\n"
                    f"\n--- Top {K} Final Results (Via Matryoshka Cascade) ---\n"
                )
                print(header_cascade, end="")
                f_log.write(header_cascade)

                query_coarse = database_embeddings[dim_coarse][query_idx : query_idx + 1]
                _, coarse_indices = indices_faiss[dim_coarse].search(query_coarse, top_n_candidates)
                candidate_idxs = coarse_indices[0]

                query_fine = database_embeddings[dim_fine][query_idx]
                candidates_fine_vectors = database_embeddings[dim_fine][candidate_idxs]
                
                scores = np.dot(candidates_fine_vectors, query_fine)
                sorted_indices = np.argsort(scores)[::-1]
                
                rank_count = 0
                for idx in sorted_indices:
                    best_faiss_idx = candidate_idxs[idx]
                    shape_id = idx_to_shape_id[best_faiss_idx]
                    score = scores[idx]
                    
                    if shape_id == shape_id_query:
                        continue
                    
                    rank_count += 1
                    result_line = f"[{rank_count}] Score: {score:.4f} | ID: {shape_id}\n"
                    print(result_line, end="")
                    
                    # 1. Writes standard formatted log
                    f_log.write(result_line)
                    # 2. Writes ONLY the raw ID string into the second file
                    f_ids.write(f"{shape_id}\n")
                    
                    if rank_count == K: 
                        break

            elif option == "2":
                header_isolated = (
                    f"\n========================================\n"
                    f"QUERY: {shape_id_query} | STRATEGY: Isolated Dimensions Evaluation\n"
                    f"========================================\n"
                )
                print(header_isolated, end="")
                f_log.write(header_isolated)

                for dim in nesting_dims:
                    query_emb = database_embeddings[dim][query_idx : query_idx + 1]
                    distances, indices = indices_faiss[dim].search(query_emb, K + 1)

                    dim_header = f"\n--- Results for Isolated Dimension: {dim} ---\n"
                    print(dim_header, end="")
                    f_log.write(dim_header)

                    rank_count = 0
                    for rank in range(K + 1):
                        faiss_idx = indices[0][rank]
                        score = distances[0][rank]
                        shape_id = idx_to_shape_id[faiss_idx]

                        if shape_id == shape_id_query:
                            continue
                        
                        rank_count += 1
                        result_line = f"[{rank_count}] Score: {score:.4f} | ID: {shape_id}\n"
                        print(result_line, end="")
                        
                        # 1. Writes standard formatted log
                        f_log.write(result_line)
                        # 2. Writes ONLY the raw ID string into the second file
                        f_ids.write(f"{shape_id}\n")
                        
                        if rank_count == K: 
                            break
            else:
                print("Invalid option. Please try again.")


if __name__ == "__main__":
    cli_args, extras = parse_args(sys.argv[1:])
    config = OmegaConf.load(cli_args.config)
    
    output_file = "retrieval_results.txt"
    for i, arg in enumerate(sys.argv):
        if arg in ["--output", "-o"] and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            break

    run_interactive_3d_retrieval(config, output_filename=output_file)