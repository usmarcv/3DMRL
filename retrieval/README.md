# Retrieval pipeline

Three stages, run in order. All commands run from the repo root, using the
`mrlclip` conda env:

```bash
conda activate mrlclip
```

## 1. Extract shape embeddings

Runs a trained checkpoint over the Objaverse-LVIS split and writes the
per-shape embeddings used by the query scripts. **Skip this step if
`retrieval/embeddings/` already has the h5/json files** — it only needs to
be re-run when the checkpoint changes.

```bash
# 3DMRL (Matryoshka) embeddings -> retrieval/embeddings/mrl_shape_embeddings.h5
torchrun --nproc_per_node=1 retrieval/extract_embeddings_3dmrl.py --config configs/retrieval/retrieval.yaml

# OpenShape baseline embeddings -> retrieval/embeddings/shape_embeddings_openshape.h5
torchrun --nproc_per_node=1 retrieval/extract_embeddings_openshape.py --config configs/retrieval/retrieval.yaml
```

Both scripts build a `DistributedSampler`, so they need a process group
initialized (`torchrun`) even for a single process. Output paths come from
the `embeddings:` block in `configs/retrieval/retrieval.yaml` — no manual
renaming needed afterwards.

## 2. Query

Interactive CLIs that search the embeddings built in step 1 — each one
prompts you for a query (an object ID for 3D-to-3D, a text description for
text-to-3D, an image path for image-to-3D) in a loop until you type `exit`.

```bash
# 3D shape -> 3D shape (Matryoshka cascade or per-dimension search)
python retrieval/query_3d_to_3d.py --config configs/retrieval/retrieval.yaml -o retrieval/results/3d/<query_name>.txt

# text -> 3D shape (3DMRL embeddings)
python retrieval/query_text_to_3d.py --config configs/retrieval/retrieval.yaml

# text -> 3D shape (OpenShape baseline embeddings)
python retrieval/query_text_to_3d_openshape.py --config configs/retrieval/retrieval.yaml

# image -> 3D shape (3DMRL embeddings)
python retrieval/query_image_to_3d.py --config configs/retrieval/retrieval.yaml

# image -> 3D shape (OpenShape baseline embeddings)
python retrieval/query_image_to_3d_openshape.py --config configs/retrieval/retrieval.yaml
```

`query_3d_to_3d.py` writes two files per run: `<query_name>.txt` (full log
with scores) and `<query_name>_ids.txt` (bare UID list, one per line) — the
latter is what step 3 consumes directly. The text-query and image-query
scripts only print results to the terminal; they don't save anything to
disk.

The image scripts embed the query with the same OpenCLIP image tower
(`ViT-bigG-14` / `laion2b_s39b_b160k`) used to build the shape embeddings, so
they land in the same joint space as the text queries — point either one at
any local image file (`.jpg`, `.png`, …) when prompted.

## 3. Download the matched objects

Resolves the UIDs from a `*_ids.txt` file against Objaverse-LVIS and
downloads the `.glb` meshes.

```bash
python retrieval/download_objects.py \
  --caminho_txt retrieval/results/3d/<query_name>_ids.txt \
  --versioned_path objaverse/<query_name>
```

`--caminho_txt` is required (there's no default — pass the `_ids.txt` file
from step 2). `--versioned_path` is where the `.glb` meshes and LVIS
metadata get downloaded to.
