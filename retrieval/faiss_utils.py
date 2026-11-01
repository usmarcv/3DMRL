"""Shared helpers for the retrieval query scripts.

Factored out of the near-identical FAISS-index-building and OpenCLIP
text-encoder-loading code that used to be copy-pasted across
``query_3d_to_3d.py``, ``query_text_to_3d.py`` and
``query_text_to_3d_openshape.py``.
"""

import json
import logging

import faiss
import numpy as np


def build_index(embeddings, normalize=True):
    """L2-normalize (optional) and build a flat inner-product FAISS index.

    Returns ``(index, embeddings)`` — the (possibly normalized) embeddings
    are returned too since callers often need them again (e.g. for the
    Matryoshka cascade re-ranking step).
    """
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))

    if normalize:
        faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings


def load_id_mapping(json_path):
    """Load ``shape_model_to_idx`` and derive the reverse ``idx_to_shape_id``."""
    with open(json_path, "r") as f:
        shape_model_to_idx = json.load(f)

    idx_to_shape_id = {v: k for k, v in shape_model_to_idx.items()}

    return shape_model_to_idx, idx_to_shape_id


def load_openclip_text_encoder(
    device,
    model_name="ViT-bigG-14",
    pretrained="laion2b_s39b_b160k",
):
    """Load the OpenCLIP text tower used to embed text queries.

    Native dimensionality of ViT-bigG-14 is 1280, matching the fixed shape
    embedding dimension produced by the extraction scripts.
    """
    import open_clip

    logging.info("Loading OpenCLIP...")

    text_model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision="fp16",
    )

    tokenizer = open_clip.get_tokenizer(model_name)
    text_model = text_model.to(device).eval()

    return text_model, tokenizer


def load_openclip_image_encoder(
    device,
    model_name="ViT-bigG-14",
    pretrained="laion2b_s39b_b160k",
):
    """Load the OpenCLIP image tower used to embed image queries.

    Same checkpoint as ``load_openclip_text_encoder`` — the shape embeddings
    were produced against this exact CLIP joint space, so the image and text
    towers must come from the same model/pretrained pair to land in it.
    Native dimensionality of ViT-bigG-14 is 1280.
    """
    import open_clip

    logging.info("Loading OpenCLIP...")

    image_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        precision="fp16",
    )

    image_model = image_model.to(device).eval()

    return image_model, preprocess
