import os
import pickle

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
from torch import nn
import torch.nn.functional as F

class GatherLayer(torch.autograd.Function):
    """
    Junta os tensores de todas as GPUs mantendo o fluxo de gradientes (backward pass).
    """
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

def gather_features(features):
    if dist.is_available() and dist.is_initialized():
        gathered_features = GatherLayer.apply(features)
        return torch.cat(gathered_features, dim=0)
    return features


class MRLProjectionHeads(nn.Module):
    """
    Uma Linear independente por granularidade MRL.

    Para cada dim em mrl_dims:
        head_dim : Linear(dim, clip_dim, bias=False)
        entrada  : z[:, :dim]  — prefixo do embedding 3D (PointBERT, 1280 dims)
        saída    : ℝ^clip_dim  — projetado no espaço do professor CLIP (1280 dims)
    """

    def __init__(self, mrl_dims, clip_dim=1280):
        super().__init__()
        self.mrl_dims = mrl_dims
        self.clip_dim = clip_dim
        self.heads = nn.ModuleList([
            nn.Linear(dim, clip_dim, bias=False)
            for dim in mrl_dims
        ])
        self._init_weights()

    def _init_weights(self):
        for head in self.heads:
            nn.init.orthogonal_(head.weight)

    def forward(self, z):
        """
        z : (B, D) — embedding do PointBERT (D = 1280)
        Retorna lista de (B, clip_dim), cada um L2-normalizado.
        """
        return [
            F.normalize(head(z[:, :dim]), dim=-1)
            for head, dim in zip(self.heads, self.mrl_dims)
        ]


def merge_results_dist(part_logits, part_labels):

    part_logits = torch.cat(part_logits, dim=0)
    part_labels = torch.cat(part_labels, dim=0)

    if not dist.is_available() or not dist.is_initialized():
        return part_logits.cpu(), part_labels.cpu()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logits_list = [torch.zeros_like(part_logits) for _ in range(world_size)]
    labels_list = [torch.zeros_like(part_labels) for _ in range(world_size)]

    dist.all_gather(logits_list, part_logits)
    dist.all_gather(labels_list, part_labels)

    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    if rank == 0:
        return logits_all.cpu(), labels_all.cpu()
    else:
        return None, None



def merge_two_branch_results_dist(part_image_logits, part_text_logits, part_labels):

    part_image_logits = torch.cat(part_image_logits, dim=0)
    part_text_logits = torch.cat(part_text_logits, dim=0)
    part_labels = torch.cat(part_labels, dim=0)

    gather_image = [torch.zeros_like(part_image_logits) for _ in range(dist.get_world_size())]
    gather_text = [torch.zeros_like(part_text_logits) for _ in range(dist.get_world_size())]
    gather_labels = [torch.zeros_like(part_labels) for _ in range(dist.get_world_size())]

    dist.all_gather(gather_image, part_image_logits)
    dist.all_gather(gather_text, part_text_logits)
    dist.all_gather(gather_labels, part_labels)

    logits_image = torch.cat(gather_image, dim=0)
    logits_text = torch.cat(gather_text, dim=0)
    labels_all = torch.cat(gather_labels, dim=0)

    return logits_image, logits_text, labels_all

