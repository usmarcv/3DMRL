import logging
import os
import math
import numpy as np
from collections import defaultdict

# import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from trainers.trainer_utils import merge_results_dist


# ──────────────────────────────────────────────────────────────────────────────
# MRLProjectionHeads — igual ao mrl_trainer.py
# ──────────────────────────────────────────────────────────────────────────────

class MRLProjectionHeads(nn.Module):
    """
    Uma Linear independente por granularidade MRL.

    Para cada dim em mrl_dims:
        head_dim : Linear(dim, clip_dim, bias=False)
        entrada  : z[:, :dim]  — prefixo do embedding 3D (PointBERT, 512 dims)
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
        z : (B, D) — embedding do PointBERT (D = 512)
        Retorna lista de (B, clip_dim), cada um L2-normalizado.
        """
        return [
            F.normalize(head(z[:, :dim]), dim=-1)
            for head, dim in zip(self.heads, self.mrl_dims)
        ]


# ──────────────────────────────────────────────────────────────────────────────
# Trainer_MRL
# ──────────────────────────────────────────────────────────────────────────────

class Trainer_MRL(object):
    """
    OpenShape + MRL + alignment adapters (image e text).

    Pipeline:
        PointBERT → z ∈ ℝ^512
            ├─► image_alignment_adapter(z) → pc_image_feat  (branch imagem)
            ├─► text_alignment_adapter(z)  → pc_text_feat   (branch texto)
            └─► MRLProjectionHeads:
                    para cada dim: head_dim(z[:,:dim]) → ℝ^1280
                    contrastive_loss(shape_proj, clip_frozen)

    Treináveis : PointBERT + MRLProjectionHeads + image/text_alignment_adapter
    Frozen     : CLIP text encoder + CLIP image encoder
    """

    def __init__(
        self,
        rank,
        config,
        model,
        logit_scale,
        image_proj,
        text_proj,
        mrl_heads,                   # MRLProjectionHeads
        optimizer,
        scheduler,
        train_loader,
        image_alignment_adapter,
        text_alignment_adapter,
        modelnet40_loader,
        objaverse_lvis_loader,
        scanobjectnn_loader,
    ):
        self.rank                    = rank
        self.config                  = config
        self.model                   = model
        self.logit_scale             = logit_scale
        self.image_proj              = image_proj
        self.text_proj               = text_proj
        self.mrl_heads               = mrl_heads
        self.optimizer               = optimizer
        self.scheduler               = scheduler
        self.train_loader            = train_loader
        self.image_alignment_adapter = image_alignment_adapter
        self.text_alignment_adapter  = text_alignment_adapter
        self.modelnet40_loader       = modelnet40_loader
        self.objaverse_lvis_loader   = objaverse_lvis_loader
        self.scanobjectnn_loader     = scanobjectnn_loader

        self.epoch = 0
        self.step  = 0
        self.best_img_contras_acc        = 0
        self.best_text_contras_acc       = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc   = 0
        self.best_lvis_acc               = 0
        self.config.ngpu = dist.get_world_size()

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _get_module(self, module):
        return module.module if hasattr(module, "module") else module

    def _get_shape_embedding(self, feat_pc, dim=None):
        """
        feat_pc : (B, 512) — saída do PointBERT
        dim     : granularidade desejada. None = head completa (maior dim).
        Retorna (B, 1280) normalizado.
        """
        heads    = self._get_module(self.mrl_heads)
        mrl_dims = heads.mrl_dims

        if dim is None:
            head = heads.heads[-1]
            z    = feat_pc[:, :mrl_dims[-1]]
        else:
            idx  = mrl_dims.index(dim)
            head = heads.heads[idx]
            z    = feat_pc[:, :dim]

        return F.normalize(head(z), dim=-1)  # (B, 1280)

    def _prepare_clip_text(self, loader):
        """Extrai, opcionalmente projeta e normaliza clip_cat_feat do dataset."""
        feat = torch.from_numpy(loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            feat = self.text_proj(feat)
        feat = F.normalize(feat, dim=-1)
        loader.dataset._clip_text_feat_normalized = feat
        return feat

    # ──────────────────────────────────────────────────────────────────────
    # MRL Loss
    # ──────────────────────────────────────────────────────────────────────

    def mrl_loss(self, feat_pc, feat_clip, logit_scale=1, mask=None, lambdas=None):
        """
        feat_pc   : (B, 512)  — saída do PointBERT (ou do adapter, se two_branch)
        feat_clip : (B, 1280) — embedding CLIP frozen (text ou image)

        Para cada dim:
            s = normalize(head_dim(feat_pc[:, :dim]))  (B, 1280)
            t = normalize(feat_clip)                   (B, 1280)
            loss += λ_dim * contrastive_loss(s, t)
        """
        heads    = self._get_module(self.mrl_heads)
        mrl_dims = heads.mrl_dims

        if lambdas is None:
            lambdas = [1.0] * len(mrl_dims)

        projected = heads(feat_pc)                    # lista de (B, 1280)
        t         = F.normalize(feat_clip, dim=-1)    # (B, 1280)

        total_loss   = 0.0
        total_acc    = 0.0
        loss_per_dim = {}
        acc_per_dim  = {}

        for i, (s, dim) in enumerate(zip(projected, mrl_dims)):
            if self.config.ngpu > 1:
                all_s  = torch.cat(torch.distributed.nn.all_gather(s), dim=0)
                all_t  = torch.cat(torch.distributed.nn.all_gather(t), dim=0)
                logits = logit_scale * (all_s @ all_t.T)
            else:
                logits = logit_scale * (s @ t.T)

            if mask is not None:
                if mask.dtype == torch.bool:
                    logits = logits.masked_fill(~mask, -1e9)
                else:
                    logits = logits + (1.0 - mask.float()) * -1e9

            labels   = torch.arange(logits.shape[0], device=self.config.device)
            loss_dim = (
                F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)
            ) / 2.0
            acc_dim  = (logits.argmax(dim=1) == labels).float().mean()

            total_loss += lambdas[i] * loss_dim
            total_acc  += acc_dim

            loss_per_dim[dim] = loss_dim.detach().item()
            acc_per_dim[dim]  = acc_dim.detach().item()

        return total_loss, total_acc / len(mrl_dims), loss_per_dim, acc_per_dim

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────────────────────────────────

    def save_model(self, name):
        torch.save(
            {
                "state_dict":                  self.model.state_dict(),
                "logit_scale":                 self.logit_scale.state_dict(),
                "mrl_heads":                   self.mrl_heads.state_dict(),
                "image_alignment_adapter":     self.image_alignment_adapter.state_dict(),
                "text_alignment_adapter":      self.text_alignment_adapter.state_dict(),
                "text_proj":                   self.text_proj.state_dict() if self.config.training.use_text_proj  else None,
                "image_proj":                  self.image_proj.state_dict() if self.config.training.use_image_proj else None,
                "optimizer":                   self.optimizer.state_dict(),
                "scheduler":                   self.scheduler.state_dict() if not self.config.training.use_openclip_optimizer_scheduler else None,
                "epoch":                       self.epoch,
                "step":                        self.step,
                "best_img_contras_acc":        self.best_img_contras_acc,
                "best_text_contras_acc":       self.best_text_contras_acc,
                "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
                "best_modelnet40_class_acc":   self.best_modelnet40_class_acc,
                "best_lvis_acc":               self.best_lvis_acc,
            },
            os.path.join(self.config.ckpt_dir, f"{name}.pt"),
        )

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.logit_scale.load_state_dict(checkpoint["logit_scale"])
        self.mrl_heads.load_state_dict(checkpoint["mrl_heads"])
        self.image_alignment_adapter.load_state_dict(checkpoint["image_alignment_adapter"])
        self.text_alignment_adapter.load_state_dict(checkpoint["text_alignment_adapter"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if not self.config.training.use_openclip_optimizer_scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.epoch = checkpoint["epoch"]
        self.step  = checkpoint["step"]
        logging.info(f"Loaded checkpoint from {path}")
        logging.info(f"  Epoch: {self.epoch}  Step: {self.step}")

    # ──────────────────────────────────────────────────────────────────────
    # Train
    # ──────────────────────────────────────────────────────────────────────

    def train_one_epoch(self):
        self.model.train()
        self.mrl_heads.train()
        self.image_alignment_adapter.train()
        self.text_alignment_adapter.train()

        if self.config.training.use_text_proj:
            self.text_proj.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()

        text_contras_acc_list = []
        img_contras_acc_list  = []

        epoch_img_loss_dim = defaultdict(list)
        epoch_img_acc_dim  = defaultdict(list)
        epoch_txt_loss_dim = defaultdict(list)
        epoch_txt_acc_dim  = defaultdict(list)

        lambdas = getattr(self.config.mrl, "lambdas", None)

        # ── Mask para hard negatives (opcional) ─────────────────────────
        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            # import numpy as np
            mask1       = np.eye(k * s).astype(bool)
            mask2       = np.kron(np.eye(s), np.ones((k, k))).astype(bool)
            mask_other  = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

        for data in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            loss = 0.0

            # ── Forward backbone 3D ──────────────────────────────────
            if not self.config.model.get("use_dense", False):
                pred_feat = self.model(
                    data["xyz"], data["features"],
                    device=self.config.device,
                    quantization_size=self.config.model.voxel_size,
                )
            else:
                pred_feat = self.model(
                    data["xyz_dense"].to(self.config.device),
                    data["features_dense"].to(self.config.device),
                )

            logit_scale = self.logit_scale(None)

            # ── Features CLIP (frozen) ───────────────────────────────
            idx       = data["has_text_idx"]
            text_feat = torch.vstack(data["text_feat"]).to(self.config.device)  # (B_text, 1280)
            img_feat  = torch.vstack(data["img_feat"]).to(self.config.device)   # (B, num_imgs*1280)

            mask = None
            if self.config.training.use_mask:
                img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
                mask = torch.logical_or(
                    torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold,
                    mask_other,
                ).detach()

            # ── Adapters: projeta pred_feat para os dois branches ────
            # pc_image_feat e pc_text_feat têm o mesmo dim de pred_feat (512),
            # mas aprenderam projeções específicas para cada modalidade.
            # O MRL fatia ESSES embeddings — forçando cada branch a organizar
            # hierarquicamente a informação relevante para sua modalidade.
            if self.config.training.mlp_type is not None:
                pc_image_feat = self.image_alignment_adapter(pred_feat)  # (B, 512)
                pc_text_feat  = self.text_alignment_adapter(pred_feat)   # (B, 512)
            else:
                # Sem adapter: usa pred_feat direto em ambos os branches
                pc_image_feat = pred_feat
                pc_text_feat  = pred_feat

            # ── Image contrastive (two_branch) ───────────────────────
            image_acc_mean = 0.0
            if self.config.training.loss_type == "two_branch" and self.config.dataset.num_imgs > 0:
                img_acc_accum = 0.0
                for i in range(self.config.dataset.num_imgs):
                    single_img = img_feat[
                        :, i * self.config.clip_embed_dim : (i + 1) * self.config.clip_embed_dim
                    ]  # (B, 1280)

                    if self.config.training.use_image_proj:
                        single_img = self.image_proj(single_img)

                    img_loss, img_acc, img_loss_dims, img_acc_dims = self.mrl_loss(
                        pc_image_feat,   # (B, 512) — branch imagem
                        single_img,      # (B, 1280) — CLIP frozen
                        logit_scale=logit_scale,
                        mask=mask,
                        lambdas=lambdas,
                    )

                    loss += img_loss * self.config.training.lambda_img_contras
                    img_acc_accum += img_acc.item()

                    for d, v in img_loss_dims.items():
                        epoch_img_loss_dim[d].append(v)
                    for d, v in img_acc_dims.items():
                        epoch_img_acc_dim[d].append(v)

                image_acc_mean = img_acc_accum / self.config.dataset.num_imgs

            # ── Text contrastive (two_branch) ────────────────────────
            text_acc_mean = 0.0
            if self.config.training.loss_type == "two_branch" and len(idx) > 0:
                if self.config.training.use_text_proj:
                    text_feat = self.text_proj(text_feat)

                txt_loss, txt_acc, txt_loss_dims, txt_acc_dims = self.mrl_loss(
                    pc_text_feat[idx],   # (B_text, 512) — branch texto
                    text_feat,           # (B_text, 1280) — CLIP frozen
                    logit_scale=logit_scale,
                    mask=mask,
                    lambdas=lambdas,
                )

                loss += txt_loss * self.config.training.lambda_text_contras
                text_acc_mean = txt_acc.item()

                for d, v in txt_loss_dims.items():
                    epoch_txt_loss_dim[d].append(v)
                for d, v in txt_acc_dims.items():
                    epoch_txt_acc_dim[d].append(v)

            # ── Acumula ──────────────────────────────────────────────
            if self.config.dataset.num_imgs > 0:
                img_contras_acc_list.append(image_acc_mean)
            if len(idx) > 0:
                text_contras_acc_list.append(text_acc_mean)

            # ── Backward ─────────────────────────────────────────────
            loss.backward()
            self.optimizer.step()

            if self.config.training.scheduler in ("cosine", "const"):
                self.scheduler(self.step)
            else:
                self.scheduler.step()

        # ── Logging por granularidade (rank 0) ───────────────────────
        if self.rank == 0:
            logging.info(
                "Train avg — text_acc: {:.4f}  img_acc: {:.4f}".format(
                    np.mean(text_contras_acc_list) if text_contras_acc_list  else 0,
                    np.mean(img_contras_acc_list) if img_contras_acc_list  else 0,
                )
            )
            header = f"{'Dim':<6} | {'Img Loss':<10} | {'Img Acc':<10} | {'Txt Loss':<10} | {'Txt Acc':<10}"
            sep    = "-" * len(header)
            logging.info(sep)
            logging.info(header)
            logging.info(sep)
            for d in sorted(set(epoch_img_loss_dim) | set(epoch_txt_loss_dim)):
                logging.info(
                    f"{d:<6} | "
                    f"{np.mean(epoch_img_loss_dim.get(d, [0])):.4f}     | "
                    f"{np.mean(epoch_img_acc_dim.get(d,  [0])):.4f}     | "
                    f"{np.mean(epoch_txt_loss_dim.get(d, [0])):.4f}     | "
                    f"{np.mean(epoch_txt_acc_dim.get(d,  [0])):.4f}"
                )
            logging.info(sep)

    # ──────────────────────────────────────────────────────────────────────
    # Train loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info(f"Epoch: {self.epoch}")
            self.train_one_epoch()
            if epoch > self.config.training.test_epoch:
                self.test_modelnet40()
                self.test_objaverse_lvis()
                self.test_scanobjectnn()
            if self.rank == 0:
                self.save_model("latest")
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model(f"epoch_{self.epoch}")

    # ──────────────────────────────────────────────────────────────────────
    # Accuracy helper
    # ──────────────────────────────────────────────────────────────────────

    def accuracy(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk       = max(topk)
            batch_size = target.size(0)
            _, pred    = output.topk(maxk, 1, True, True)
            pred       = pred.t()
            correct    = pred.eq(target.reshape(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct

    # ──────────────────────────────────────────────────────────────────────
    # Zero-shot tests — loop genérico
    # ──────────────────────────────────────────────────────────────────────

    def _run_zero_shot(self, loader):
        """
        Inferência zero-shot usando a head de maior granularidade.
        clip_text_feat_normalized deve ter sido setado por _prepare_clip_text.
        """
        clip_text_norm = loader.dataset._clip_text_feat_normalized
        logits_all     = []
        labels_all     = []

        with torch.no_grad():
            for data in loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(
                        data["xyz"], data["features"],
                        device=self.config.device,
                        quantization_size=self.config.model.voxel_size,
                    )
                else:
                    pred_feat = self.model(data["xyz_dense"], data["features_dense"])

                # Usa a head completa para máxima qualidade zero-shot.
                # Para retrieval em escala, substitua por dim=64 (shortlisting)
                # seguido de dim=None (re-ranking).
                shape_emb = self._get_shape_embedding(pred_feat)   # (B, 1280)
                logits    = shape_emb @ clip_text_norm.T
                labels    = data["category"].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

        return merge_results_dist(logits_all, labels_all)

    def test_modelnet40(self):
        self.model.eval()
        self.mrl_heads.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()

        self._prepare_clip_text(self.modelnet40_loader)
        logits_all, labels_all = self._run_zero_shot(self.modelnet40_loader)

        if self.rank == 0:
            n = len(self.modelnet40_loader.dataset)
            logits_all, labels_all = logits_all[:n], labels_all[:n]

            topk_acc, _     = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))
            per_cat_correct = torch.zeros(40, device=self.config.device)
            per_cat_count   = torch.zeros(40, device=self.config.device)

            for i in range(40):
                idx = labels_all == i
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i]   = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            class_acc   = (per_cat_correct / per_cat_count).mean()

            if overall_acc > self.best_modelnet40_overall_acc:
                self.best_modelnet40_overall_acc = overall_acc
                self.save_model("best_modelnet40")
            if class_acc > self.best_modelnet40_class_acc:
                self.best_modelnet40_class_acc = class_acc

            logging.info(
                f"ModelNet40 — overall: {overall_acc:.4f} ({self.best_modelnet40_overall_acc:.4f}) "
                f"class: {class_acc:.4f} ({self.best_modelnet40_class_acc:.4f}) "
                f"top1/3/5: {topk_acc[0].item():.2f}/{topk_acc[1].item():.2f}/{topk_acc[2].item():.2f}"
            )
            torch.save(
                {"logits": logits_all, "labels": labels_all,
                 "overall_acc": overall_acc, "class_acc": class_acc},
                os.path.join(self.config.ckpt_dir, f"modelnet40_epoch_{self.epoch}.pth"),
            )

    def test_objaverse_lvis(self):
        self.model.eval()
        self.mrl_heads.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()

        self._prepare_clip_text(self.objaverse_lvis_loader)
        logits_all, labels_all = self._run_zero_shot(self.objaverse_lvis_loader)

        if self.rank == 0:
            n = len(self.objaverse_lvis_loader.dataset)
            logits_all, labels_all = logits_all[:n], labels_all[:n]

            topk_acc, _     = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))
            per_cat_correct = torch.zeros(1156, device=self.config.device)
            per_cat_count   = torch.zeros(1156, device=self.config.device)

            for i in torch.unique(labels_all):
                idx = labels_all == i
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i]   = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            class_acc   = (per_cat_correct / per_cat_count).mean()

            if overall_acc > self.best_lvis_acc:
                self.best_lvis_acc = overall_acc
                self.save_model("best_lvis")

            logging.info(
                f"ObjaverseLVIS — overall: {overall_acc:.4f} ({self.best_lvis_acc:.4f}) "
                f"class: {class_acc:.4f} "
                f"top1/3/5: {topk_acc[0].item():.2f}/{topk_acc[1].item():.2f}/{topk_acc[2].item():.2f}"
            )
            torch.save(
                {"logits": logits_all, "labels": labels_all,
                 "overall_acc": overall_acc, "class_acc": class_acc},
                os.path.join(self.config.ckpt_dir, f"objaverse_lvis_epoch_{self.epoch}.pth"),
            )

    def test_scanobjectnn(self):
        self.model.eval()
        self.mrl_heads.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()

        self._prepare_clip_text(self.scanobjectnn_loader)
        logits_all, labels_all = self._run_zero_shot(self.scanobjectnn_loader)

        if self.rank == 0:
            n = len(self.scanobjectnn_loader.dataset)
            logits_all, labels_all = logits_all[:n], labels_all[:n]

            topk_acc, _     = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))
            per_cat_correct = torch.zeros(15, device=self.config.device)
            per_cat_count   = torch.zeros(15, device=self.config.device)

            for i in range(15):
                idx = labels_all == i
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i]   = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            class_acc   = (per_cat_correct / per_cat_count).mean()

            logging.info(
                f"ScanObjectNN — overall: {overall_acc:.4f}  class: {class_acc:.4f} "
                f"top1/3/5: {topk_acc[0].item():.2f}/{topk_acc[1].item():.2f}/{topk_acc[2].item():.2f}"
            )
            torch.save(
                {"logits": logits_all, "labels": labels_all,
                 "overall_acc": overall_acc, "class_acc": class_acc},
                os.path.join(self.config.ckpt_dir, f"scanobjectnn_epoch_{self.epoch}.pth"),
            )
