import logging
import os
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from trainers.trainer_utils import merge_results_dist


# ──────────────────────────────────────────────────────────────────────────────
# MRL Projection Heads
# ──────────────────────────────────────────────────────────────────────────────

class MRLProjectionHeads(nn.Module):
    """
    Uma Linear independente por granularidade MRL.

    Para cada dim em mrl_dims:
        head_dim : Linear(dim, clip_dim, bias=False)
        entrada  : z[:, :dim]  — prefixo do embedding 3D
        saída    : ℝ^clip_dim  — projetado no espaço do professor CLIP

    Por que heads independentes e não uma submatriz compartilhada?
    - A submatriz compartilhada (weight[:, :dim]) força a mesma direção de
      projeção para todos os prefixos. Heads independentes permitem que cada
      granularidade aprenda a projeção ótima para aquele número de dimensões.
    - É mais fiel ao paper MRL (Seção 3.1), que usa classificadores independentes
      para cada granularidade durante o treino.

    Parâmetros:
        mrl_dims  : lista de granularidades, ex: [8, 16, 32, 64, 128, 256, 512]
        clip_dim  : dimensão do espaço professor (1280 para OpenCLIP ViT-L)
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
        """
        Inicialização ortogonal — estabiliza o treino nos prefixos menores,
        que têm sinal mais fraco por terem menos dimensões de entrada.
        """
        for head in self.heads:
            nn.init.orthogonal_(head.weight)

    def forward(self, z):
        """
        z : (B, D) — embedding completo do PointBERT (D = 512 no seu caso)

        Retorna lista de tensores normalizados, um por granularidade:
            [(B, clip_dim), (B, clip_dim), ...]
        Cada tensor já está L2-normalizado — pronto para logits contrastivos.
        """
        return [
            F.normalize(head(z[:, :dim]), dim=-1)
            for head, dim in zip(self.heads, self.mrl_dims)
        ]


# ──────────────────────────────────────────────────────────────────────────────
# MRL Trainer
# ──────────────────────────────────────────────────────────────────────────────

class MRL_Trainer(object):
    """
    OpenShape + Matryoshka Representation Learning (MRL).

    Pipeline:
        PointBERT → z ∈ ℝ^512
            └─► MRLProjectionHeads:
                    para cada dim em [8, 16, 32, 64, 128, 256, 512]:
                        head_dim(z[:, :dim]) → ℝ^1280  (normalizado)
                        contrastive_loss(shape_proj, clip_frozen)

    O PointBERT é forçado a organizar informação hierarquicamente:
    os primeiros 8 dims devem capturar o conceito mais grosseiro,
    512 dims capturam a representação completa.

    Treináveis : PointBERT + MRLProjectionHeads
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
        mrl_heads,          # instância de MRLProjectionHeads
        optimizer,
        scheduler,
        train_loader,
        modelnet40_loader,
        objaverse_lvis_loader,
        scanobjectnn_loader,
    ):
        self.rank         = rank
        self.config       = config
        self.model        = model
        self.logit_scale  = logit_scale
        self.image_proj   = image_proj
        self.text_proj    = text_proj
        self.mrl_heads    = mrl_heads
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader     = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader   = scanobjectnn_loader

        self.epoch = 0
        self.step  = 0
        self.best_img_contras_acc        = 0
        self.best_text_contras_acc       = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc   = 0
        self.best_lvis_acc               = 0
        self.config.ngpu = dist.get_world_size()

    # ──────────────────────────────────────────────────────────────────────
    # Helper
    # ──────────────────────────────────────────────────────────────────────

    def _get_module(self, module):
        return module.module if hasattr(module, "module") else module

    # ──────────────────────────────────────────────────────────────────────
    # MRL Loss
    # ──────────────────────────────────────────────────────────────────────

    def mrl_loss(self, feat_pc, feat_clip, logit_scale=1, mask=None, lambdas=None):
        """
        feat_pc   : (B, 512)  — saída do PointBERT, SEM normalização
        feat_clip : (B, 1280) — embedding CLIP frozen (text ou image)
        logit_scale: escalar (já extraído do módulo)
        lambdas   : pesos por granularidade. None = uniforme (fiel ao paper MRL)

        Para cada granularidade dim:
            s = normalize(head_dim(feat_pc[:, :dim]))   (B, 1280)
            t = normalize(feat_clip)                    (B, 1280)
            loss += λ_dim * contrastive_loss(s, t)

        Por que comparar sempre em ℝ^1280?
            O "professor" (CLIP) só existe em 1280 dims. O aluno aprende a
            comprimir dim dims de informação 3D em uma representação que seja
            útil nesse espaço — cada head aprende a projeção ótima para aquele
            número de dimensões de entrada.
        """
        heads   = self._get_module(self.mrl_heads)
        mrl_dims = heads.mrl_dims

        if lambdas is None:
            lambdas = [1.0] * len(mrl_dims)

        # Projeta todas as granularidades de uma vez — evita recomputar prefixos
        # projected: lista de (B, 1280), cada uma já normalizada
        projected = heads(feat_pc)

        # Normaliza CLIP uma única vez fora do loop
        t = F.normalize(feat_clip, dim=-1)  # (B, 1280)

        total_loss   = None
        total_acc    = None
        loss_per_dim = {}
        acc_per_dim  = {}

        for i, (s, dim) in enumerate(zip(projected, mrl_dims)):

            if self.config.ngpu > 1:
                all_s = torch.cat(torch.distributed.nn.all_gather(s), dim=0)   # (B*G, 1280)
                all_t = torch.cat(torch.distributed.nn.all_gather(t), dim=0)   # (B*G, 1280)
                logits = logit_scale * (all_s @ all_t.T)                        # (B*G, B*G)

                # Labels globais — corretos para o loss
                labels = torch.arange(logits.shape[0], device=self.config.device)
                loss_dim = (
                    F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)
                ) / 2.0

                # Acc apenas na fatia local deste rank
                B_local     = s.shape[0]
                local_start = self.rank * B_local
                local_end   = local_start + B_local

                local_logits = logits[local_start:local_end]          # (B, B*G)
                local_labels = torch.arange(
                    local_start, local_end, device=self.config.device
                )                                                      # positivos globais
                acc_dim = (local_logits.argmax(dim=1) == local_labels).float().mean()

            else:
                logits   = logit_scale * (s @ t.T)
                labels   = torch.arange(logits.shape[0], device=self.config.device)
                loss_dim = (
                    F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)
                ) / 2.0
                acc_dim  = (logits.argmax(dim=1) == labels).float().mean()

            # total_loss += lambdas[i] * loss_dim
            total_loss = total_loss / sum(lambdas)
            total_acc  += acc_dim

            loss_per_dim[dim] = loss_dim.detach().item()
            acc_per_dim[dim]  = acc_dim.detach().item()

        return total_loss, total_acc / len(mrl_dims), loss_per_dim, acc_per_dim

    # ──────────────────────────────────────────────────────────────────────
    # Inference helper
    # ──────────────────────────────────────────────────────────────────────

    def _get_shape_embedding(self, feat_pc, dim=None):
        """
        Projeção para inferência zero-shot.

        feat_pc : (B, 512) — saída do PointBERT
        dim     : granularidade desejada. None = usa a head completa (maior dim).

        Retorna embedding (B, 1280) normalizado, pronto para similaridade coseno
        com embeddings CLIP de texto/imagem.

        Para retrieval eficiente em escala, passe dim menor (ex: dim=64) para
        shortlisting e dim=None para re-ranking.
        """
        heads    = self._get_module(self.mrl_heads)
        mrl_dims = heads.mrl_dims

        if dim is None:
            # Head da maior granularidade — melhor qualidade
            head = heads.heads[-1]
            z    = feat_pc[:, :mrl_dims[-1]]
        else:
            # Head da granularidade solicitada
            idx  = mrl_dims.index(dim)
            head = heads.heads[idx]
            z    = feat_pc[:, :dim]

        return F.normalize(head(z), dim=-1)  # (B, 1280)

    # ──────────────────────────────────────────────────────────────────────
    # Train
    # ──────────────────────────────────────────────────────────────────────

    def train_one_epoch(self):
        self.model.train()
        self.mrl_heads.train()

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

        # Lambdas opcionais por granularidade — None = uniforme
        lambdas = getattr(self.config.mrl, "lambdas", None)

        for data in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            loss = None
            
            # ── Forward do backbone 3D ───────────────────────────────
            # pred_feat: (B, 512) — CLS token do PointBERT, sem projeção
            if not self.config.model.get("use_dense", False):
                pred_feat = self.model(
                    data["xyz"], data["features"],
                    device=self.config.device,
                    quantization_size=self.config.model.voxel_size,
                )
            else:
                pred_feat = self.model(data["xyz_dense"], data["features_dense"])

            logit_scale = self.logit_scale(None)

            # ── Features CLIP (frozen) ───────────────────────────────
            idx       = data["has_text_idx"]
            text_feat = torch.vstack(data["text_feat"]).to(self.config.device)  # (B_text, 1280)
            img_feat  = torch.vstack(data["img_feat"]).to(self.config.device)   # (B, num_imgs*1280)

            # ── Image contrastive ────────────────────────────────────
            image_acc_mean = 0.0
            if self.config.dataset.num_imgs > 0:
                img_acc_accum = 0.0
                for i in range(self.config.dataset.num_imgs):
                    single_img = img_feat[
                        :, i * self.config.clip_embed_dim : (i + 1) * self.config.clip_embed_dim
                    ]  # (B, 1280)

                    if self.config.training.use_image_proj:
                        single_img = self.image_proj(single_img)

                    img_loss, img_acc, img_loss_dims, img_acc_dims = self.mrl_loss(
                        pred_feat,   # (B, 512)
                        single_img,  # (B, 1280)
                        logit_scale=logit_scale,
                        lambdas=lambdas,
                    )

                    loss += img_loss * self.config.training.lambda_img_contras
                    img_acc_accum += img_acc.item()

                    for d, v in img_loss_dims.items():
                        epoch_img_loss_dim[d].append(v)
                    for d, v in img_acc_dims.items():
                        epoch_img_acc_dim[d].append(v)

                image_acc_mean = img_acc_accum / self.config.dataset.num_imgs

            # ── Text contrastive ─────────────────────────────────────
            text_acc_mean = 0.0
            if len(idx) > 0:
                if self.config.training.use_text_proj:
                    text_feat = self.text_proj(text_feat)

                txt_loss, txt_acc, txt_loss_dims, txt_acc_dims = self.mrl_loss(
                    pred_feat[idx],  # (B_text, 512)
                    text_feat,       # (B_text, 1280)
                    logit_scale=logit_scale,
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
                    np.mean(text_contras_acc_list) if text_contras_acc_list else 0,
                    np.mean(img_contras_acc_list)  if img_contras_acc_list  else 0,
                )
            )
            header = f"{'Dim':<6} | {'Img Loss':<10} | {'Img Acc':<10} | {'Txt Loss':<10} | {'Txt Acc':<10}"
            sep    = "-" * len(header)
            logging.info(sep)
            logging.info(header)
            logging.info(sep)
            all_dims = sorted(set(epoch_img_loss_dim) | set(epoch_txt_loss_dim))
            for d in all_dims:
                logging.info(
                    f"{d:<6} | "
                    f"{np.mean(epoch_img_loss_dim.get(d, [0])):.4f}     | "
                    f"{np.mean(epoch_img_acc_dim.get(d,  [0])):.4f}     | "
                    f"{np.mean(epoch_txt_loss_dim.get(d, [0])):.4f}     | "
                    f"{np.mean(epoch_txt_acc_dim.get(d,  [0])):.4f}"
                )
            logging.info(sep)

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ──────────────────────────────────────────────────────────────────────

    def save_model(self, name):
        torch.save(
            {
                "state_dict":                  self.model.state_dict(),
                "logit_scale":                 self.logit_scale.state_dict(),
                "mrl_heads":                   self.mrl_heads.state_dict(),
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
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if not self.config.training.use_openclip_optimizer_scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.epoch = checkpoint["epoch"]
        self.step  = checkpoint["step"]
        logging.info(f"Loaded checkpoint from {path}")
        logging.info(f"  Epoch: {self.epoch}  Step: {self.step}")

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
    # Zero-shot tests
    # ──────────────────────────────────────────────────────────────────────

    def _run_zero_shot(self, loader):
        """
        Loop genérico de inferência zero-shot.
        Usa a head de maior granularidade (máxima qualidade) por padrão.
        """
        logits_all = []
        labels_all = []

        # clip_text_feat já foi normalizado pelo caller
        clip_text_feat = loader.dataset._clip_text_feat_normalized

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

                shape_emb = self._get_shape_embedding(pred_feat)          # (B, 1280)
                logits    = shape_emb @ clip_text_feat.T
                labels    = data["category"].to(self.config.device)

                logits_all.append(logits.detach())
                labels_all.append(labels)

        return merge_results_dist(logits_all, labels_all)

    def _prepare_clip_text(self, loader):
        """Extrai e normaliza clip_cat_feat do dataset, com text_proj opcional."""
        feat = torch.from_numpy(loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            feat = self.text_proj(feat)
        feat = F.normalize(feat, dim=-1)
        # Guarda no dataset para o _run_zero_shot acessar
        loader.dataset._clip_text_feat_normalized = feat
        return feat

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


# ──────────────────────────────────────────────────────────────────────────────
# Como instanciar no seu script principal
# ──────────────────────────────────────────────────────────────────────────────
#
# mrl_heads = MRLProjectionHeads(
#     mrl_dims = config.mrl.dims,   # ex: [8, 16, 32, 64, 128, 256, 512]
#     clip_dim = config.clip_embed_dim,  # 1280
# ).to(device)
#
# optimizer = torch.optim.AdamW([
#     {"params": model.parameters()},
#     {"params": mrl_heads.parameters(), "lr": config.training.lr * 2},
#     # lr maior para as heads pois elas começam do zero (encoder tem pretrain)
# ], lr=config.training.lr, weight_decay=config.training.wd)
#
# trainer = MRL_Trainer(
#     rank=rank,
#     config=config,
#     model=model,
#     logit_scale=logit_scale,
#     image_proj=image_proj,
#     text_proj=text_proj,
#     mrl_heads=mrl_heads,          # <-- substituiu mrl_nested_proj
#     optimizer=optimizer,
#     scheduler=scheduler,
#     train_loader=train_loader,
#     modelnet40_loader=modelnet40_loader,
#     objaverse_lvis_loader=objaverse_lvis_loader,
#     scanobjectnn_loader=scanobjectnn_loader,
# )
#
# Configuração YAML necessária:
# mrl:
#   dims: [8, 16, 32, 64, 128, 256, 512]
#   lambdas: null   # null = uniforme; ou [1,1,1,1,1,1,2] para dar mais peso ao full


# import logging
# import math
# import os
# from collections import defaultdict

# import numpy as np
# import torch
# import torch.distributed as dist
# import torch.distributed.nn
# import torch.nn.functional as F
# from tqdm import tqdm

# from trainers.trainer_utils import merge_results_dist


# class MRL_Trainer(object):
#     """
#     OpenShape + Matryoshka Representation Learning (MRL) — sem adapters.

#     Pipeline fiel ao paper MRL:
#         PointBERT → (B, 512)
#             └─► mrl_nested_proj Linear(512, 1280, bias=False)
#                     Para cada dim em [8, 16, 32, 64, 128, 256, 512]:
#                         shape[:,:dim] @ weight[:,:dim].T → (B, 1280) → [:dim] → (B, dim)
#                         clip[:,:dim]                                           → (B, dim)
#                         contrastive_loss(shape_dim, clip_dim)

#     O PointBERT é forçado a organizar informação hierarquicamente nos 512 dims.
#     CLIP text/image: completamente frozen, fatiados direto.
#     Treináveis: PointBERT + mrl_nested_proj.
#     """

#     def __init__(
#         self,
#         rank,
#         config,
#         model,
#         logit_scale,
#         image_proj,
#         text_proj,
#         mrl_image_proj,
#         mrl_text_proj,
#         mrl_nested_proj,
#         optimizer,
#         scheduler,
#         train_loader,
#         modelnet40_loader,
#         objaverse_lvis_loader,
#         scanobjectnn_loader,
#     ):
#         self.rank           = rank
#         self.config         = config
#         self.model          = model
#         self.logit_scale    = logit_scale
#         self.image_proj     = image_proj
#         self.text_proj      = text_proj
#         self.mrl_image_proj = mrl_image_proj
#         self.mrl_text_proj  = mrl_text_proj
#         self.mrl_nested_proj = mrl_nested_proj
#         self.optimizer      = optimizer
#         self.scheduler      = scheduler
#         self.train_loader   = train_loader
#         self.modelnet40_loader     = modelnet40_loader
#         self.objaverse_lvis_loader = objaverse_lvis_loader
#         self.scanobjectnn_loader   = scanobjectnn_loader

#         self.epoch = 0
#         self.step  = 0
#         self.best_img_contras_acc        = 0
#         self.best_text_contras_acc       = 0
#         self.best_modelnet40_overall_acc = 0
#         self.best_modelnet40_class_acc   = 0
#         self.best_lvis_acc               = 0
#         self.config.ngpu = dist.get_world_size()

#     # ──────────────────────────────────────────────────────────────────────
#     # Helper
#     # ──────────────────────────────────────────────────────────────────────

#     def _get_module(self, module):
#         """Desempacota DDP se necessário."""
#         return module.module if hasattr(module, "module") else module

#     # ──────────────────────────────────────────────────────────────────────
#     # MRL Loss — sem adapters, fiel ao paper
#     # ──────────────────────────────────────────────────────────────────────

#     # def mrl_loss(self, feat_pc, feat_clip, mrl_dims, logit_scale=1, mask=None):
#     #     """
#     #     feat_pc   : (B, 512)  — saída raw do PointBERT
#     #     feat_clip : (B, 1280) — embedding CLIP frozen
#     #     mrl_dims  : [8, 16, 32, 64, 128, 256, 512]

#     #     Para cada dim:
#     #         1. shape[:,:dim] projetado via mrl_nested_proj fatiado → (B, 1280) → fatia [:dim]
#     #         2. clip[:,:dim] fatiado direto
#     #         3. contrastive loss simétrica entre (B, dim) e (B, dim)

#     #     Pesos uniformes — fiel ao paper MRL.
#     #     """
#     #     total_loss   = 0.0
#     #     total_acc    = 0.0
#     #     loss_per_dim = {}
#     #     acc_per_dim  = {}

#     #     # Pesos uniformes — fiel ao paper MRL
#     #     weights = torch.ones(len(mrl_dims), device=self.config.device)

#     #     # Weight matrix: (1280, 512)
#     #     full_weight    = self._get_module(self.mrl_nested_proj).weight

#     #     # Normaliza clip uma vez fora do loop
#     #     feat_clip_norm = F.normalize(feat_clip, dim=-1)  # (B, 1280)

#     #     for i, dim in enumerate(mrl_dims):

#     #         # 1. Fatia shape e projeta via submatriz de mrl_nested_proj
#     #         #    weight[:, :dim]: (1280, dim)
#     #         #    F.linear(x, W) = x @ W.T → (B, dim) @ (dim, 1280) = (B, 1280)
#     #         feat_sliced    = feat_pc[:, :dim]                    # (B, dim)
#     #         sliced_weight  = full_weight[:, :dim]                # (1280, dim)
#     #         feat_projected = F.linear(feat_sliced, sliced_weight) # (B, 1280)

#     #         # 2. Fatia as primeiras `dim` dims de ambos
#     #         s = F.normalize(feat_projected[:, :dim], dim=-1)    # (B, dim)
#     #         # t = F.normalize(feat_clip_norm[:, :dim], dim=-1)    # (B, dim)
#     #         # t = feat_clip_norm

#     #         # 3. Logits
#     #         if self.config.ngpu > 1:
#     #             all_s  = torch.cat(torch.distributed.nn.all_gather(s), dim=0)
#     #             all_t  = torch.cat(torch.distributed.nn.all_gather(feat_clip_norm), dim=0)
#     #             logits = logit_scale * (all_s @ all_t.T)
#     #         else:
#     #             logits = logit_scale * (s @ t.T)  # (B, B)

#     #         # 4. Mask
#     #         if mask is not None:
#     #             if mask.dtype == torch.bool:
#     #                 logits = logits.masked_fill(~mask, -1e9)
#     #             else:
#     #                 logits = logits + (1.0 - mask.float()) * -1e9

#     #         # 5. Contrastive loss simétrica
#     #         labels   = torch.arange(logits.shape[0], device=self.config.device)
#     #         loss_dim = (
#     #             F.cross_entropy(logits, labels) +
#     #             F.cross_entropy(logits.T, labels)
#     #         ) / 2.0
#     #         acc_dim  = (logits.argmax(dim=1) == labels).float().mean()

#     #         total_loss += weights[i] * loss_dim
#     #         total_acc  += acc_dim

#     #         loss_per_dim[dim] = loss_dim.detach().item()
#     #         acc_per_dim[dim]  = acc_dim.detach().item()

#     #     return total_loss, total_acc / len(mrl_dims), loss_per_dim, acc_per_dim

#     def mrl_loss(self, feat_pc, feat_clip, mrl_dims, logit_scale=1, mask=None):
        
#         total_loss   = 0.0
#         total_acc    = 0.0
#         loss_per_dim = {}
#         acc_per_dim  = {}

#         weights = torch.ones(len(mrl_dims), device=self.config.device)
#         full_weight = self._get_module(self.mrl_nested_proj).weight

#         # Normaliza clip UMA VEZ fora do loop (B, 1280)
#         feat_clip_norm = F.normalize(feat_clip, dim=-1)  

#         for i, dim in enumerate(mrl_dims):

#             # 1. Fatia shape e projeta via submatriz
#             feat_sliced    = feat_pc[:, :dim]                     # (B, dim)
#             sliced_weight  = full_weight[:, :dim]                 # (1280, dim)
            
#             # Projeta o 3D de volta para o tamanho do professor
#             feat_projected = F.linear(feat_sliced, sliced_weight) # (B, 1280)

#             # 2. NORMALIZA TUDO NO TAMANHO DO PROFESSOR (1280)
#             # NÃO fatie o feat_projected e NÃO fatie o feat_clip_norm!
#             s = F.normalize(feat_projected, dim=-1)               # (B, 1280)
#             t = feat_clip_norm                                    # (B, 1280)

#             # 3. Logits (O combate justo de 1280 vs 1280)
#             if self.config.ngpu > 1:
#                 # Nota: assumindo que seu all_gather mantém o autograd!
#                 all_s  = torch.cat(torch.distributed.nn.all_gather(s), dim=0)
#                 all_t  = torch.cat(torch.distributed.nn.all_gather(t), dim=0)
#                 logits = logit_scale * (all_s @ all_t.T)
#             else:
#                 logits = logit_scale * (s @ t.T)  # (B, B)

#             # 4. Mask
#             if mask is not None:
#                 if mask.dtype == torch.bool:
#                     logits = logits.masked_fill(~mask, -1e9)
#                 else:
#                     logits = logits + (1.0 - mask.float()) * -1e9

#             # 5. Contrastive loss simétrica
#             labels   = torch.arange(logits.shape[0], device=self.config.device)
#             loss_dim = (
#                 F.cross_entropy(logits, labels) +
#                 F.cross_entropy(logits.T, labels)
#             ) / 2.0
            
#             acc_dim  = (logits.argmax(dim=1) == labels).float().mean()

#             total_loss += weights[i] * loss_dim
#             total_acc  += acc_dim

#             loss_per_dim[dim] = loss_dim.detach().item()
#             acc_per_dim[dim]  = acc_dim.detach().item()

#         return total_loss, total_acc / len(mrl_dims), loss_per_dim, acc_per_dim


#     # ──────────────────────────────────────────────────────────────────────
#     # Inference helper
#     # ──────────────────────────────────────────────────────────────────────

#     def _get_shape_embedding(self, pred_feat):
#         """
#         Projeção completa para inferência zero-shot.
#         (B, 512) → mrl_nested_proj completo → (B, 1280), normalizado.
#         """
#         weight = self._get_module(self.mrl_nested_proj).weight  # (1280, 512)
#         return F.normalize(F.linear(pred_feat, weight), dim=-1)  # (B, 1280)

#     # ──────────────────────────────────────────────────────────────────────
#     # Train
#     # ──────────────────────────────────────────────────────────────────────

#     def train_one_epoch(self):
#         self.model.train()
#         self.mrl_nested_proj.train()

#         if self.config.training.use_text_proj:
#             self.text_proj.train()
#         if self.config.training.use_image_proj:
#             self.image_proj.train()

#         text_contras_acc_list = []
#         img_contras_acc_list  = []

#         epoch_img_loss_dim = defaultdict(list)
#         epoch_img_acc_dim  = defaultdict(list)
#         epoch_txt_loss_dim = defaultdict(list)
#         epoch_txt_acc_dim  = defaultdict(list)

#         for data in tqdm(self.train_loader):
#             self.step += 1
#             self.optimizer.zero_grad()
#             loss = 0.0

#             # ── 3D backbone forward ──────────────────────────────────
#             # pred_feat: (B, 512) — CLS token do PointBERT, SEM projeção
#             if not self.config.model.get("use_dense", False):
#                 pred_feat = self.model(
#                     data["xyz"], data["features"],
#                     device=self.config.device,
#                     quantization_size=self.config.model.voxel_size,
#                 )
#             else:
#                 pred_feat = self.model(data["xyz_dense"], data["features_dense"])

#             logit_scale = self.logit_scale(None)

#             # ── Coleta features CLIP ─────────────────────────────────
#             idx       = data["has_text_idx"]
#             text_feat = torch.vstack(data["text_feat"]).to(self.config.device)  # (B_text, 1280)
#             img_feat  = torch.vstack(data["img_feat"]).to(self.config.device)   # (B, num_imgs*1280)

#             mask = None

#             # ── Image contrastive ────────────────────────────────────
#             image_acc_mean = 0.0
#             if self.config.dataset.num_imgs > 0:
#                 img_acc_accum = 0.0
#                 for i in range(self.config.dataset.num_imgs):
#                     single_image_feat = img_feat[
#                         :, i * self.config.clip_embed_dim : (i + 1) * self.config.clip_embed_dim
#                     ]  # (B, 1280)

#                     if self.config.training.use_image_proj:
#                         single_image_feat = self.image_proj(single_image_feat)

#                     img_contras_loss, img_contras_acc, \
#                     img_loss_dims, img_acc_dims = self.mrl_loss(
#                         pred_feat,          # (B, 512)
#                         single_image_feat,  # (B, 1280)
#                         self.config.mrl.dims,
#                         logit_scale=logit_scale,
#                         mask=mask,
#                     )

#                     loss += img_contras_loss * self.config.training.lambda_img_contras
#                     img_acc_accum += img_contras_acc.item()

#                     for d, val in img_loss_dims.items():
#                         epoch_img_loss_dim[d].append(val)
#                     for d, val in img_acc_dims.items():
#                         epoch_img_acc_dim[d].append(val)

#                 image_acc_mean = img_acc_accum / self.config.dataset.num_imgs

#             # ── Text contrastive ─────────────────────────────────────
#             text_acc_mean = 0.0
#             if len(idx) > 0:
#                 if self.config.training.use_text_proj:
#                     text_feat = self.text_proj(text_feat)

#                 text_contras_loss, text_contras_acc, \
#                 txt_loss_dims, txt_acc_dims = self.mrl_loss(
#                     pred_feat[idx],  # (B_text, 512)
#                     text_feat,       # (B_text, 1280)
#                     self.config.mrl.dims,
#                     logit_scale=logit_scale,
#                     mask=mask,
#                 )

#                 loss += text_contras_loss * self.config.training.lambda_text_contras
#                 text_acc_mean = text_contras_acc.item()

#                 for d, val in txt_loss_dims.items():
#                     epoch_txt_loss_dim[d].append(val)
#                 for d, val in txt_acc_dims.items():
#                     epoch_txt_acc_dim[d].append(val)

#             # ── Acumula ──────────────────────────────────────────────
#             if self.config.dataset.num_imgs > 0:
#                 img_contras_acc_list.append(image_acc_mean)
#             if len(idx) > 0:
#                 text_contras_acc_list.append(text_acc_mean)

#             # ── Backward ─────────────────────────────────────────────
#             loss.backward()
#             self.optimizer.step()

#             if self.config.training.scheduler in ("cosine", "const"):
#                 self.scheduler(self.step)
#             else:
#                 self.scheduler.step()

#         # ── Logging (rank 0) ─────────────────────────────────────────
#         if self.rank == 0:
#             logging.info(
#                 "Train avg: text_contras_acc: {0} image_contras_acc: {1}".format(
#                     np.mean(text_contras_acc_list) if text_contras_acc_list else 0,
#                     np.mean(img_contras_acc_list)  if img_contras_acc_list  else 0,
#                 )
#             )

#             header = f"{'Dim':<6} | {'Img Loss':<10} | {'Img Acc':<10} | {'Txt Loss':<10} | {'Txt Acc':<10}"
#             logging.info("-" * len(header))
#             logging.info(header)
#             logging.info("-" * len(header))

#             all_dims = sorted(set(epoch_img_loss_dim.keys()) | set(epoch_txt_loss_dim.keys()))
#             for d in all_dims:
#                 i_loss = np.mean(epoch_img_loss_dim.get(d, [0]))
#                 i_acc  = np.mean(epoch_img_acc_dim.get(d,  [0]))
#                 t_loss = np.mean(epoch_txt_loss_dim.get(d, [0]))
#                 t_acc  = np.mean(epoch_txt_acc_dim.get(d,  [0]))
#                 logging.info(
#                     f"{d:<6} | {i_loss:.4f}     | {i_acc:.4f}     | {t_loss:.4f}     | {t_acc:.4f}"
#                 )
#             logging.info("-" * len(header))

#     # ──────────────────────────────────────────────────────────────────────
#     # Checkpoint
#     # ──────────────────────────────────────────────────────────────────────

#     def save_model(self, name):
#         torch.save(
#             {
#                 "state_dict":                   self.model.state_dict(),
#                 "logit_scale":                  self.logit_scale.state_dict(),
#                 "mrl_nested_proj":              self.mrl_nested_proj.state_dict(),
#                 "text_proj":                    self.text_proj.state_dict() if self.config.training.use_text_proj else None,
#                 "image_proj":                   self.image_proj.state_dict() if self.config.training.use_image_proj else None,
#                 "optimizer":                    self.optimizer.state_dict(),
#                 "scheduler":                    self.scheduler.state_dict() if not self.config.training.use_openclip_optimizer_scheduler else None,
#                 "epoch":                        self.epoch,
#                 "step":                         self.step,
#                 "best_img_contras_acc":         self.best_img_contras_acc,
#                 "best_text_contras_acc":        self.best_text_contras_acc,
#                 "best_modelnet40_overall_acc":  self.best_modelnet40_overall_acc,
#                 "best_modelnet40_class_acc":    self.best_modelnet40_class_acc,
#                 "best_lvis_acc":                self.best_lvis_acc,
#             },
#             os.path.join(self.config.ckpt_dir, "{}.pt".format(name)),
#         )

#     def load_from_checkpoint(self, path):
#         checkpoint = torch.load(path)
#         self.model.load_state_dict(checkpoint["state_dict"])
#         self.logit_scale.load_state_dict(checkpoint["logit_scale"])
#         self.mrl_nested_proj.load_state_dict(checkpoint["mrl_nested_proj"])
#         self.optimizer.load_state_dict(checkpoint["optimizer"])
#         if not self.config.training.use_openclip_optimizer_scheduler:
#             self.scheduler.load_state_dict(checkpoint["scheduler"])
#         self.epoch = checkpoint["epoch"]
#         self.step  = checkpoint["step"]
#         logging.info("Loaded checkpoint from {}".format(path))
#         logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

#     # ──────────────────────────────────────────────────────────────────────
#     # Train loop
#     # ──────────────────────────────────────────────────────────────────────

#     def train(self):
#         for epoch in range(self.epoch, self.config.training.max_epoch):
#             self.epoch = epoch
#             if self.rank == 0:
#                 logging.info("Epoch: {}".format(self.epoch))
#             self.train_one_epoch()
#             if epoch > self.config.training.test_epoch:
#                 self.test_modelnet40()
#                 self.test_objaverse_lvis()
#                 self.test_scanobjectnn()
#             if self.rank == 0:
#                 self.save_model("latest")
#             if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
#                 self.save_model("epoch_{}".format(self.epoch))

#     # ──────────────────────────────────────────────────────────────────────
#     # Accuracy helper
#     # ──────────────────────────────────────────────────────────────────────

#     def accuracy(self, output, target, topk=(1,)):
#         with torch.no_grad():
#             maxk       = max(topk)
#             batch_size = target.size(0)
#             _, pred    = output.topk(maxk, 1, True, True)
#             pred       = pred.t()
#             correct    = pred.eq(target.reshape(1, -1).expand_as(pred))
#             res = []
#             for k in topk:
#                 correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#                 res.append(correct_k.mul_(100.0 / batch_size))
#             return res, correct

#     # ──────────────────────────────────────────────────────────────────────
#     # Zero-shot tests
#     # ──────────────────────────────────────────────────────────────────────

#     def test_modelnet40(self):
#         self.model.eval()
#         if self.config.training.use_text_proj:
#             self.text_proj.eval()

#         clip_text_feat = torch.from_numpy(
#             self.modelnet40_loader.dataset.clip_cat_feat
#         ).to(self.config.device)
#         if self.config.training.use_text_proj:
#             clip_text_feat = self.text_proj(clip_text_feat)

#         logits_all = []
#         labels_all = []
#         with torch.no_grad():
#             for data in self.modelnet40_loader:
#                 if not self.config.model.get("use_dense", False):
#                     pred_feat = self.model(
#                         data["xyz"], data["features"],
#                         device=self.config.device,
#                         quantization_size=self.config.model.voxel_size,
#                     )
#                 else:
#                     pred_feat = self.model(data["xyz_dense"], data["features_dense"])

#                 shape_emb = self._get_shape_embedding(pred_feat)          # (B, 1280)
#                 logits    = shape_emb @ F.normalize(clip_text_feat, dim=-1).T
#                 labels    = data["category"].to(self.config.device)
#                 logits_all.append(logits.detach())
#                 labels_all.append(labels)

#         logits_all, labels_all = merge_results_dist(logits_all, labels_all)

#         if self.rank == 0:
#             dataset_size = len(self.modelnet40_loader.dataset)
#             logits_all   = logits_all[:dataset_size]
#             labels_all   = labels_all[:dataset_size]

#             topk_acc, _     = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))
#             per_cat_correct = torch.zeros(40).to(self.config.device)
#             per_cat_count   = torch.zeros(40).to(self.config.device)
#             for i in range(40):
#                 idx = labels_all == i
#                 if idx.sum() > 0:
#                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
#                     per_cat_count[i]   = idx.sum()

#             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
#             per_cat_acc = per_cat_correct / per_cat_count

#             if overall_acc > self.best_modelnet40_overall_acc:
#                 self.best_modelnet40_overall_acc = overall_acc
#             if per_cat_acc.mean() > self.best_modelnet40_class_acc:
#                 self.best_modelnet40_class_acc = per_cat_acc.mean()

#             logging.info(
#                 "Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})".format(
#                     overall_acc, self.best_modelnet40_overall_acc,
#                     per_cat_acc.mean(), self.best_modelnet40_class_acc,
#                 )
#             )
#             logging.info(
#                 "Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}".format(
#                     topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
#                 )
#             )
#             torch.save(
#                 {"logits": logits_all, "labels": labels_all,
#                  "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
#                 os.path.join(self.config.ckpt_dir, f"modelnet40_epoch_{self.epoch}.pth"),
#             )

#     def test_objaverse_lvis(self):
#         self.model.eval()
#         if self.config.training.use_text_proj:
#             self.text_proj.eval()

#         clip_text_feat = torch.from_numpy(
#             self.objaverse_lvis_loader.dataset.clip_cat_feat
#         ).to(self.config.device)
#         if self.config.training.use_text_proj:
#             clip_text_feat = self.text_proj(clip_text_feat)

#         per_cat_correct = torch.zeros(1156).to(self.config.device)
#         per_cat_count   = torch.zeros(1156).to(self.config.device)

#         logits_all = []
#         labels_all = []
#         with torch.no_grad():
#             for data in self.objaverse_lvis_loader:
#                 if not self.config.model.get("use_dense", False):
#                     pred_feat = self.model(
#                         data["xyz"], data["features"],
#                         device=self.config.device,
#                         quantization_size=self.config.model.voxel_size,
#                     )
#                 else:
#                     pred_feat = self.model(data["xyz_dense"], data["features_dense"])

#                 shape_emb = self._get_shape_embedding(pred_feat)
#                 logits    = shape_emb @ F.normalize(clip_text_feat, dim=-1).T
#                 labels    = data["category"].to(self.config.device)
#                 logits_all.append(logits.detach())
#                 labels_all.append(labels)

#         logits_all, labels_all = merge_results_dist(logits_all, labels_all)

#         if self.rank == 0:
#             dataset_size = len(self.objaverse_lvis_loader.dataset)
#             logits_all   = logits_all[:dataset_size]
#             labels_all   = labels_all[:dataset_size]

#             topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))

#             for i in torch.unique(labels_all):
#                 idx = labels_all == i
#                 if idx.sum() > 0:
#                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
#                     per_cat_count[i]   = idx.sum()

#             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
#             per_cat_acc = per_cat_correct / per_cat_count

#             if overall_acc > self.best_lvis_acc:
#                 self.best_lvis_acc = overall_acc

#             logging.info(
#                 "Test ObjaverseLVIS: overall acc: {0} class_acc: {1}".format(
#                     overall_acc, per_cat_acc.mean()
#                 )
#             )
#             logging.info(
#                 "Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}".format(
#                     topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
#                 )
#             )
#             torch.save(
#                 {"logits": logits_all, "labels": labels_all,
#                  "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
#                 os.path.join(self.config.ckpt_dir, f"objaverse_lvis_epoch_{self.epoch}.pth"),
#             )

#     def test_scanobjectnn(self):
#         self.model.eval()
#         if self.config.training.use_text_proj:
#             self.text_proj.eval()

#         clip_text_feat = torch.from_numpy(
#             self.scanobjectnn_loader.dataset.clip_cat_feat
#         ).to(self.config.device)
#         if self.config.training.use_text_proj:
#             clip_text_feat = self.text_proj(clip_text_feat)

#         per_cat_correct = torch.zeros(15).to(self.config.device)
#         per_cat_count   = torch.zeros(15).to(self.config.device)

#         logits_all = []
#         labels_all = []
#         with torch.no_grad():
#             for data in self.scanobjectnn_loader:
#                 if not self.config.model.get("use_dense", False):
#                     pred_feat = self.model(
#                         data["xyz"], data["features"],
#                         device=self.config.device,
#                         quantization_size=self.config.model.voxel_size,
#                     )
#                 else:
#                     pred_feat = self.model(data["xyz_dense"], data["features_dense"])

#                 shape_emb = self._get_shape_embedding(pred_feat)
#                 logits    = shape_emb @ F.normalize(clip_text_feat, dim=-1).T
#                 labels    = data["category"].to(self.config.device)
#                 logits_all.append(logits.detach())
#                 labels_all.append(labels)

#         logits_all, labels_all = merge_results_dist(logits_all, labels_all)

#         if self.rank == 0:
#             dataset_size = len(self.scanobjectnn_loader.dataset)
#             logits_all   = logits_all[:dataset_size]
#             labels_all   = labels_all[:dataset_size]

#             topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))

#             for i in range(15):
#                 idx = labels_all == i
#                 if idx.sum() > 0:
#                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
#                     per_cat_count[i]   = idx.sum()

#             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
#             per_cat_acc = per_cat_correct / per_cat_count

#             logging.info(
#                 "Test ScanObjectNN: overall acc: {0} class_acc: {1}".format(
#                     overall_acc, per_cat_acc.mean()
#                 )
#             )
#             logging.info(
#                 "Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}".format(
#                     topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
#                 )
#             )
#             torch.save(
#                 {"logits": logits_all, "labels": labels_all,
#                  "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
#                 os.path.join(self.config.ckpt_dir, f"scanobjectnn_epoch_{self.epoch}.pth"),
#             )

# # import logging
# # import os
# # from collections import defaultdict

# # import numpy as np
# # import torch
# # import torch.distributed as dist
# # import torch.distributed.nn
# # import torch.nn.functional as F
# # from tqdm import tqdm

# # from trainers.trainer_utils import merge_results_dist


# # class MRL_Trainer(object):
# #     """
# #     OpenShape + Matryoshka Representation Learning (MRL) — versão limpa.

# #     Pipeline:
# #         PointBERT (512) → Linear(512→1280) → (B, 1280)
# #             └─► mrl_loss fatia [:dim] em shape e clip diretamente
# #                     shape[:,:dim] vs clip[:,:dim] — sem projeção extra

# #     Vantagens desta abordagem:
# #         - Consistência perfeita entre treino e inferência
# #         - Fiel ao paper MRL — só fatia, sem submatriz nem adapters
# #         - mrl_nested_proj removido — Linear(512→1280) fica dentro do backbone
# #         - mrl_loss mais simples e direta

# #     Treináveis: PointBERT (inclui Linear(512→1280) interno)
# #     Frozen: CLIP text encoder, CLIP image encoder
# #     """

# #     def __init__(
# #         self,
# #         rank,
# #         config,
# #         model,
# #         logit_scale,
# #         image_proj,
# #         text_proj,
# #         optimizer,
# #         scheduler,
# #         train_loader,
# #         modelnet40_loader,
# #         objaverse_lvis_loader,
# #         scanobjectnn_loader,
# #     ):
# #         self.rank            = rank
# #         self.config          = config
# #         self.model           = model
# #         self.logit_scale     = logit_scale
# #         self.image_proj      = image_proj
# #         self.text_proj       = text_proj
# #         self.optimizer       = optimizer
# #         self.scheduler       = scheduler
# #         self.train_loader    = train_loader
# #         self.modelnet40_loader     = modelnet40_loader
# #         self.objaverse_lvis_loader = objaverse_lvis_loader
# #         self.scanobjectnn_loader   = scanobjectnn_loader

# #         self.epoch = 0
# #         self.step  = 0
# #         self.best_img_contras_acc        = 0
# #         self.best_text_contras_acc       = 0
# #         self.best_modelnet40_overall_acc = 0
# #         self.best_modelnet40_class_acc   = 0
# #         self.best_lvis_acc               = 0
# #         self.config.ngpu = dist.get_world_size()

# #     # ──────────────────────────────────────────────────────────────────────
# #     # MRL Loss — versão limpa, fiel ao paper
# #     # ──────────────────────────────────────────────────────────────────────

# #     def mrl_loss(self, feat_pc, feat_clip, mrl_dims, logit_scale=1, mask=None):
# #         """
# #         feat_pc   : (B, 1280) — saída do PointBERT já projetado para espaço CLIP
# #         feat_clip : (B, 1280) — embedding CLIP frozen (text ou image)
# #         mrl_dims  : [8, 16, 32, 64, 128, 256, 512, 1280]

# #         Para cada dim:
# #             s = normalize(feat_pc[:,:dim])    (B, dim)
# #             t = normalize(feat_clip[:,:dim])  (B, dim)
# #             contrastive_loss(s, t)

# #         Pesos uniformes — fiel ao paper MRL.
# #         Sem submatriz, sem adapters, sem projeção extra.
# #         """
# #         total_loss   = 0.0
# #         total_acc    = 0.0
# #         loss_per_dim = {}
# #         acc_per_dim  = {}

# #         # pesos uniformes — fiel ao paper MRL
# #         weights = torch.ones(len(mrl_dims), device=self.config.device)

# #         # normaliza clip uma vez fora do loop
# #         feat_clip_norm = F.normalize(feat_clip, dim=-1)  # (B, 1280)

# #         for i, dim in enumerate(mrl_dims):

# #             # fatia direto — sem projeção extra
# #             s = F.normalize(feat_pc[:, :dim], dim=-1)         # (B, dim)
# #             t = F.normalize(feat_clip_norm[:, :dim], dim=-1)  # (B, dim)

# #             # logits
# #             if self.config.ngpu > 1:
# #                 all_s  = torch.cat(torch.distributed.nn.all_gather(s), dim=0)
# #                 all_t  = torch.cat(torch.distributed.nn.all_gather(t), dim=0)
# #                 logits = logit_scale * (all_s @ all_t.T)
# #             else:
# #                 logits = logit_scale * (s @ t.T)  # (B, B)

# #             # mask
# #             if mask is not None:
# #                 if mask.dtype == torch.bool:
# #                     logits = logits.masked_fill(~mask, -1e9)
# #                 else:
# #                     logits = logits + (1.0 - mask.float()) * -1e9

# #             # contrastive loss simétrica
# #             labels   = torch.arange(logits.shape[0], device=self.config.device)
# #             loss_dim = (
# #                 F.cross_entropy(logits, labels) +
# #                 F.cross_entropy(logits.T, labels)
# #             ) / 2.0
# #             acc_dim  = (logits.argmax(dim=1) == labels).float().mean()

# #             total_loss += weights[i] * loss_dim
# #             total_acc  += acc_dim

# #             loss_per_dim[dim] = loss_dim.detach().item()
# #             acc_per_dim[dim]  = acc_dim.detach().item()

# #         return total_loss, total_acc / len(mrl_dims), loss_per_dim, acc_per_dim

# #     # ──────────────────────────────────────────────────────────────────────
# #     # Inference helper
# #     # ──────────────────────────────────────────────────────────────────────

# #     def _get_shape_embedding(self, pred_feat, dim=None):
# #         """
# #         pred_feat : (B, 1280) — saída do PointBERT
# #         dim       : dimensão desejada (None = 1280 completo)

# #         Retorna embedding normalizado (B, dim) pronto para similaridade coseno.
# #         """
# #         if dim is None:
# #             return F.normalize(pred_feat, dim=-1)           # (B, 1280)
# #         return F.normalize(pred_feat[:, :dim], dim=-1)      # (B, dim)

# #     # ──────────────────────────────────────────────────────────────────────
# #     # Train
# #     # ──────────────────────────────────────────────────────────────────────

# #     def train_one_epoch(self):
# #         self.model.train()

# #         if self.config.training.use_text_proj:
# #             self.text_proj.train()
# #         if self.config.training.use_image_proj:
# #             self.image_proj.train()

# #         text_contras_acc_list = []
# #         img_contras_acc_list  = []

# #         epoch_img_loss_dim = defaultdict(list)
# #         epoch_img_acc_dim  = defaultdict(list)
# #         epoch_txt_loss_dim = defaultdict(list)
# #         epoch_txt_acc_dim  = defaultdict(list)

# #         for data in tqdm(self.train_loader):
# #             self.step += 1
# #             self.optimizer.zero_grad()
# #             loss = 0.0

# #             # ── 3D backbone forward ──────────────────────────────────
# #             # pred_feat: (B, 1280) — PointBERT + Linear(512→1280) interno
# #             if not self.config.model.get("use_dense", False):
# #                 pred_feat = self.model(
# #                     data["xyz"], data["features"],
# #                     device=self.config.device,
# #                     quantization_size=self.config.model.voxel_size,
# #                 )
# #             else:
# #                 pred_feat = self.model(data["xyz_dense"], data["features_dense"])

# #             logit_scale = self.logit_scale(None)

# #             # ── Coleta features CLIP ─────────────────────────────────
# #             idx       = data["has_text_idx"]
# #             text_feat = torch.vstack(data["text_feat"]).to(self.config.device)  # (B_text, 1280)
# #             img_feat  = torch.vstack(data["img_feat"]).to(self.config.device)   # (B, num_imgs*1280)

# #             mask = None

# #             # ── Image contrastive ────────────────────────────────────
# #             image_acc_mean = 0.0
# #             if self.config.dataset.num_imgs > 0:
# #                 img_acc_accum = 0.0
# #                 for i in range(self.config.dataset.num_imgs):
# #                     single_image_feat = img_feat[
# #                         :, i * self.config.clip_embed_dim : (i + 1) * self.config.clip_embed_dim
# #                     ]  # (B, 1280)

# #                     if self.config.training.use_image_proj:
# #                         single_image_feat = self.image_proj(single_image_feat)

# #                     img_contras_loss, img_contras_acc, \
# #                     img_loss_dims, img_acc_dims = self.mrl_loss(
# #                         pred_feat,          # (B, 1280)
# #                         single_image_feat,  # (B, 1280)
# #                         self.config.mrl.dims,
# #                         logit_scale=logit_scale,
# #                         mask=mask,
# #                     )

# #                     loss += img_contras_loss * self.config.training.lambda_img_contras
# #                     img_acc_accum += img_contras_acc.item()

# #                     for d, val in img_loss_dims.items():
# #                         epoch_img_loss_dim[d].append(val)
# #                     for d, val in img_acc_dims.items():
# #                         epoch_img_acc_dim[d].append(val)

# #                 image_acc_mean = img_acc_accum / self.config.dataset.num_imgs

# #             # ── Text contrastive ─────────────────────────────────────
# #             text_acc_mean = 0.0
# #             if len(idx) > 0:
# #                 if self.config.training.use_text_proj:
# #                     text_feat = self.text_proj(text_feat)

# #                 text_contras_loss, text_contras_acc, \
# #                 txt_loss_dims, txt_acc_dims = self.mrl_loss(
# #                     pred_feat[idx],  # (B_text, 1280)
# #                     text_feat,       # (B_text, 1280)
# #                     self.config.mrl.dims,
# #                     logit_scale=logit_scale,
# #                     mask=mask,
# #                 )

# #                 loss += text_contras_loss * self.config.training.lambda_text_contras
# #                 text_acc_mean = text_contras_acc.item()

# #                 for d, val in txt_loss_dims.items():
# #                     epoch_txt_loss_dim[d].append(val)
# #                 for d, val in txt_acc_dims.items():
# #                     epoch_txt_acc_dim[d].append(val)

# #             # ── Acumula ──────────────────────────────────────────────
# #             if self.config.dataset.num_imgs > 0:
# #                 img_contras_acc_list.append(image_acc_mean)
# #             if len(idx) > 0:
# #                 text_contras_acc_list.append(text_acc_mean)

# #             # ── Backward ─────────────────────────────────────────────
# #             loss.backward()
# #             self.optimizer.step()

# #             if self.config.training.scheduler in ("cosine", "const"):
# #                 self.scheduler(self.step)
# #             else:
# #                 self.scheduler.step()

# #         # ── Logging (rank 0) ─────────────────────────────────────────
# #         if self.rank == 0:
# #             logging.info(
# #                 "Train avg: text_contras_acc: {0} image_contras_acc: {1}".format(
# #                     np.mean(text_contras_acc_list) if text_contras_acc_list else 0,
# #                     np.mean(img_contras_acc_list)  if img_contras_acc_list  else 0,
# #                 )
# #             )

# #             header = f"{'Dim':<6} | {'Img Loss':<10} | {'Img Acc':<10} | {'Txt Loss':<10} | {'Txt Acc':<10}"
# #             logging.info("-" * len(header))
# #             logging.info(header)
# #             logging.info("-" * len(header))

# #             all_dims = sorted(set(epoch_img_loss_dim.keys()) | set(epoch_txt_loss_dim.keys()))
# #             for d in all_dims:
# #                 i_loss = np.mean(epoch_img_loss_dim.get(d, [0]))
# #                 i_acc  = np.mean(epoch_img_acc_dim.get(d,  [0]))
# #                 t_loss = np.mean(epoch_txt_loss_dim.get(d, [0]))
# #                 t_acc  = np.mean(epoch_txt_acc_dim.get(d,  [0]))
# #                 logging.info(
# #                     f"{d:<6} | {i_loss:.4f}     | {i_acc:.4f}     | {t_loss:.4f}     | {t_acc:.4f}"
# #                 )
# #             logging.info("-" * len(header))

# #     # ──────────────────────────────────────────────────────────────────────
# #     # Checkpoint
# #     # ──────────────────────────────────────────────────────────────────────

# #     def save_model(self, name):
# #         torch.save(
# #             {
# #                 "state_dict":                   self.model.state_dict(),
# #                 "logit_scale":                  self.logit_scale.state_dict(),
# #                 "text_proj":                    self.text_proj.state_dict() if self.config.training.use_text_proj else None,
# #                 "image_proj":                   self.image_proj.state_dict() if self.config.training.use_image_proj else None,
# #                 "optimizer":                    self.optimizer.state_dict(),
# #                 "scheduler":                    self.scheduler.state_dict() if not self.config.training.use_openclip_optimizer_scheduler else None,
# #                 "epoch":                        self.epoch,
# #                 "step":                         self.step,
# #                 "best_img_contras_acc":         self.best_img_contras_acc,
# #                 "best_text_contras_acc":        self.best_text_contras_acc,
# #                 "best_modelnet40_overall_acc":  self.best_modelnet40_overall_acc,
# #                 "best_modelnet40_class_acc":    self.best_modelnet40_class_acc,
# #                 "best_lvis_acc":                self.best_lvis_acc,
# #             },
# #             os.path.join(self.config.ckpt_dir, "{}.pt".format(name)),
# #         )

# #     def load_from_checkpoint(self, path):
# #         checkpoint = torch.load(path)
# #         self.model.load_state_dict(checkpoint["state_dict"])
# #         self.logit_scale.load_state_dict(checkpoint["logit_scale"])
# #         self.optimizer.load_state_dict(checkpoint["optimizer"])
# #         if not self.config.training.use_openclip_optimizer_scheduler:
# #             self.scheduler.load_state_dict(checkpoint["scheduler"])
# #         self.epoch = checkpoint["epoch"]
# #         self.step  = checkpoint["step"]
# #         logging.info("Loaded checkpoint from {}".format(path))
# #         logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

# #     # ──────────────────────────────────────────────────────────────────────
# #     # Train loop
# #     # ──────────────────────────────────────────────────────────────────────

# #     def train(self):
# #         for epoch in range(self.epoch, self.config.training.max_epoch):
# #             self.epoch = epoch
# #             if self.rank == 0:
# #                 logging.info("Epoch: {}".format(self.epoch))
# #             self.train_one_epoch()
# #             if epoch > self.config.training.test_epoch:
# #                 self.test_modelnet40()
# #                 self.test_objaverse_lvis()
# #                 self.test_scanobjectnn()
# #             if self.rank == 0:
# #                 self.save_model("latest")
# #             if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
# #                 self.save_model("epoch_{}".format(self.epoch))

# #     # ──────────────────────────────────────────────────────────────────────
# #     # Accuracy helper
# #     # ──────────────────────────────────────────────────────────────────────

# #     def accuracy(self, output, target, topk=(1,)):
# #         with torch.no_grad():
# #             maxk       = max(topk)
# #             batch_size = target.size(0)
# #             _, pred    = output.topk(maxk, 1, True, True)
# #             pred       = pred.t()
# #             correct    = pred.eq(target.reshape(1, -1).expand_as(pred))
# #             res = []
# #             for k in topk:
# #                 correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
# #                 res.append(correct_k.mul_(100.0 / batch_size))
# #             return res, correct

# #     # ──────────────────────────────────────────────────────────────────────
# #     # Zero-shot tests
# #     # ──────────────────────────────────────────────────────────────────────

# #     def _run_zero_shot(self, loader, clip_text_feat, num_classes):
# #         """
# #         Loop de inferência zero-shot genérico.
# #         Retorna logits_all e labels_all já merged via dist.
# #         """
# #         logits_all = []
# #         labels_all = []
# #         clip_text_norm = F.normalize(clip_text_feat, dim=-1)

# #         with torch.no_grad():
# #             for data in loader:
# #                 if not self.config.model.get("use_dense", False):
# #                     pred_feat = self.model(
# #                         data["xyz"], data["features"],
# #                         device=self.config.device,
# #                         quantization_size=self.config.model.voxel_size,
# #                     )
# #                 else:
# #                     pred_feat = self.model(data["xyz_dense"], data["features_dense"])

# #                 # projeção completa (dim=1280) para zero-shot
# #                 shape_emb = self._get_shape_embedding(pred_feat)  # (B, 1280)
# #                 logits    = shape_emb @ clip_text_norm.T
# #                 labels    = data["category"].to(self.config.device)
# #                 logits_all.append(logits.detach())
# #                 labels_all.append(labels)

# #         return merge_results_dist(logits_all, labels_all)

# #     def test_modelnet40(self):
# #         self.model.eval()
# #         if self.config.training.use_text_proj:
# #             self.text_proj.eval()

# #         clip_text_feat = torch.from_numpy(
# #             self.modelnet40_loader.dataset.clip_cat_feat
# #         ).to(self.config.device)
# #         if self.config.training.use_text_proj:
# #             clip_text_feat = self.text_proj(clip_text_feat)

# #         logits_all, labels_all = self._run_zero_shot(
# #             self.modelnet40_loader, clip_text_feat, num_classes=40
# #         )

# #         if self.rank == 0:
# #             dataset_size = len(self.modelnet40_loader.dataset)
# #             logits_all   = logits_all[:dataset_size]
# #             labels_all   = labels_all[:dataset_size]

# #             topk_acc, _     = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))
# #             per_cat_correct = torch.zeros(40).to(self.config.device)
# #             per_cat_count   = torch.zeros(40).to(self.config.device)
# #             for i in range(40):
# #                 idx = labels_all == i
# #                 if idx.sum() > 0:
# #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# #                     per_cat_count[i]   = idx.sum()

# #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# #             per_cat_acc = per_cat_correct / per_cat_count

# #             if overall_acc > self.best_modelnet40_overall_acc:
# #                 self.best_modelnet40_overall_acc = overall_acc
# #             if per_cat_acc.mean() > self.best_modelnet40_class_acc:
# #                 self.best_modelnet40_class_acc = per_cat_acc.mean()

# #             logging.info(
# #                 "Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})".format(
# #                     overall_acc, self.best_modelnet40_overall_acc,
# #                     per_cat_acc.mean(), self.best_modelnet40_class_acc,
# #                 )
# #             )
# #             logging.info(
# #                 "Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}".format(
# #                     topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
# #                 )
# #             )
# #             torch.save(
# #                 {"logits": logits_all, "labels": labels_all,
# #                  "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
# #                 os.path.join(self.config.ckpt_dir, f"modelnet40_epoch_{self.epoch}.pth"),
# #             )

# #     def test_objaverse_lvis(self):
# #         self.model.eval()
# #         if self.config.training.use_text_proj:
# #             self.text_proj.eval()

# #         clip_text_feat = torch.from_numpy(
# #             self.objaverse_lvis_loader.dataset.clip_cat_feat
# #         ).to(self.config.device)
# #         if self.config.training.use_text_proj:
# #             clip_text_feat = self.text_proj(clip_text_feat)

# #         logits_all, labels_all = self._run_zero_shot(
# #             self.objaverse_lvis_loader, clip_text_feat, num_classes=1156
# #         )

# #         if self.rank == 0:
# #             dataset_size    = len(self.objaverse_lvis_loader.dataset)
# #             logits_all      = logits_all[:dataset_size]
# #             labels_all      = labels_all[:dataset_size]

# #             topk_acc, _     = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))
# #             per_cat_correct = torch.zeros(1156).to(self.config.device)
# #             per_cat_count   = torch.zeros(1156).to(self.config.device)

# #             for i in torch.unique(labels_all):
# #                 idx = labels_all == i
# #                 if idx.sum() > 0:
# #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# #                     per_cat_count[i]   = idx.sum()

# #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# #             per_cat_acc = per_cat_correct / per_cat_count

# #             if overall_acc > self.best_lvis_acc:
# #                 self.best_lvis_acc = overall_acc

# #             logging.info(
# #                 "Test ObjaverseLVIS: overall acc: {0} class_acc: {1}".format(
# #                     overall_acc, per_cat_acc.mean()
# #                 )
# #             )
# #             logging.info(
# #                 "Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}".format(
# #                     topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
# #                 )
# #             )
# #             torch.save(
# #                 {"logits": logits_all, "labels": labels_all,
# #                  "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
# #                 os.path.join(self.config.ckpt_dir, f"objaverse_lvis_epoch_{self.epoch}.pth"),
# #             )

# #     def test_scanobjectnn(self):
# #         self.model.eval()
# #         if self.config.training.use_text_proj:
# #             self.text_proj.eval()

# #         clip_text_feat = torch.from_numpy(
# #             self.scanobjectnn_loader.dataset.clip_cat_feat
# #         ).to(self.config.device)
# #         if self.config.training.use_text_proj:
# #             clip_text_feat = self.text_proj(clip_text_feat)

# #         logits_all, labels_all = self._run_zero_shot(
# #             self.scanobjectnn_loader, clip_text_feat, num_classes=15
# #         )

# #         if self.rank == 0:
# #             dataset_size    = len(self.scanobjectnn_loader.dataset)
# #             logits_all      = logits_all[:dataset_size]
# #             labels_all      = labels_all[:dataset_size]

# #             topk_acc, _     = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))
# #             per_cat_correct = torch.zeros(15).to(self.config.device)
# #             per_cat_count   = torch.zeros(15).to(self.config.device)

# #             for i in range(15):
# #                 idx = labels_all == i
# #                 if idx.sum() > 0:
# #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# #                     per_cat_count[i]   = idx.sum()

# #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# #             per_cat_acc = per_cat_correct / per_cat_count

# #             logging.info(
# #                 "Test ScanObjectNN: overall acc: {0} class_acc: {1}".format(
# #                     overall_acc, per_cat_acc.mean()
# #                 )
# #             )
# #             logging.info(
# #                 "Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}".format(
# #                     topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
# #                 )
# #             )
# #             torch.save(
# #                 {"logits": logits_all, "labels": labels_all,
# #                  "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
# #                 os.path.join(self.config.ckpt_dir, f"scanobjectnn_epoch_{self.epoch}.pth"),
# #             )