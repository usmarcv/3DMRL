# 3D-MRL Rebuttal Experiments — BMVC 2026

Experiments to address reviewer concerns for "3D-MRL: Nested Multimodal 3D Representations
via Matryoshka Representation Learning" (BMVC 2026, Submission #162).

## Experiments

### Experiment 1: Efficiency Profiling
Addresses all three reviewers' requests for deployment-efficiency measurements.

Measures at each MRL dimension (10, 20, 40, 80, 160, 320, 640, 1280):
- Per-embedding storage (bytes)
- Total embedding storage for the Objaverse-LVIS gallery (46,205 shapes)
- FAISS index disk size (MB)
- FAISS index build time (s)
- Single-query retrieval latency (ms, mean ± std)
- Batched retrieval throughput (queries/second)

Outputs:
- `results/efficiency/efficiency_table.csv` — efficiency metrics table
- `results/efficiency/efficiency_table.tex` — LaTeX version
- `results/plots/pareto_accuracy_vs_storage.png/pdf` — Accuracy vs Storage Pareto frontier

### Experiment 2: PCA and Truncation Baselines
Addresses R1 and R2's requests for stronger dimensionality reduction baselines.

Compares at matched dimensions:
1. **3D-MRL prefix** — our method, learned nested prefixes
2. **OpenShape truncated** — naive prefix truncation (no MRL training)
3. **OpenShape + PCA** — fit PCA on OpenShape embeddings, project to target dim
4. **OpenShape + random projection** — Gaussian random projection to target dim

Evaluated on Objaverse-LVIS zero-shot classification (1,156 categories).

Outputs:
- `results/zero_shot/zero_shot_lvis.csv` — accuracy per method per dimension
- `results/zero_shot/zero_shot_lvis.tex` — LaTeX table
- `results/plots/accuracy_vs_dimension_lvis.png/pdf` — 4-curve accuracy plot

## Quick Start

```bash
# Run both experiments
python rebuttal_exps/run_experiments.py \
    --mrl-h5 mrl_shape_embeddings.h5 \
    --openshape-h5 shape_embeddings_openshape.h5 \
    --lvis-csv lvis.csv \
    --lvis-text-feat data/meta_data/lvis_cat_name_pt_feat.npy \
    --mapping-json mrl_shape_model_to_idx.json \
    --experiments efficiency zero_shot \
    --devices cpu \
    --output-dir rebuttal_exps/results

# Run only efficiency
python rebuttal_exps/run_experiments.py \
    ... \
    --experiments efficiency

# Run with CUDA benchmarks
python rebuttal_exps/run_experiments.py \
    ... \
    --experiments efficiency zero_shot \
    --devices cpu cuda
```

## Requirements

- Python 3.8+
- faiss-cpu (or faiss-gpu for CUDA benchmarks)
- numpy, pandas, h5py, matplotlib
- scikit-learn (for PCA baseline)

## Expected Key Results

1. **3D-MRL preserves accuracy at low dimensions** while naive OpenShape truncation
   collapses (e.g., at 40-D, 3D-MRL achieves ~12% Top-1 while OpenShape truncated
   achieves near-random accuracy).

2. **PCA and random projection cannot recover** the nested structure — 3D-MRL
   consistently outperforms them at all dimensions below 1280.

3. **Storage-latency trade-off**: 3D-MRL at 160-D uses only 12.5% of the storage
   of 1280-D while achieving a meaningful speedup, validating the paper's core
   motivation of flexible deployment from a single model.