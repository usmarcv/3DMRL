import logging
import os

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.distributed.nn
import torch.nn.functional as F
import wandb
import torch.distributed as dist
from trainers.trainer_utils import merge_results_dist
from tqdm import tqdm
from trainers.trainer_utils import merge_two_branch_results_dist


class TrainerOpenShape(object):
    def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, train_loader, \
                 modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None, clip_adapter=None):
        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.epoch = 0
        self.step = 0
        self.clip_adapter = clip_adapter
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0
        self.config.ngpu = dist.get_world_size()

    def _cfg_get(self, obj, key, default=None):
        """Read config values from dict/OmegaConf-like objects safely."""
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        try:
            return obj.get(key, default)
        except Exception:
            return getattr(obj, key, default)

    def _get_module(self, module):
        """Return the wrapped module when using DataParallel/DDP."""
        return module.module if hasattr(module, "module") else module

    def _clean_ddp_state_dict(self, state_dict):
        """Remove the 'module.' prefix from checkpoints saved with DDP."""
        if state_dict is None:
            return None
        return {
            k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

    def _load_state_dict_clean(self, module, state_dict, strict=True, name="module"):
        """Load a state_dict robustly, handling DDP/non-DDP checkpoints."""
        state_dict = self._clean_ddp_state_dict(state_dict)
        msg = self._get_module(module).load_state_dict(state_dict, strict=strict)
        logging.info("Loaded {} state_dict: {}".format(name, msg))
        return msg

    def load_from_checkpoint(self, path, resume=True, strict=True):
        """
        Load OpenShape checkpoint.

        Use resume=False for evaluation-only compression baselines, e.g.
        OpenShape truncated and OpenShape Linear PCA. This avoids requiring optimizer
        and scheduler states when you only want to test a frozen checkpoint.
        """
        checkpoint = torch.load(path, map_location="cpu")

        self._load_state_dict_clean(
            self.model,
            checkpoint["state_dict"],
            strict=strict,
            name="model",
        )

        if "logit_scale" in checkpoint and checkpoint["logit_scale"] is not None:
            self._load_state_dict_clean(
                self.logit_scale,
                checkpoint["logit_scale"],
                strict=strict,
                name="logit_scale",
            )

        if (
            self.config.training.use_text_proj
            and "text_proj" in checkpoint
            and checkpoint["text_proj"] is not None
        ):
            self._load_state_dict_clean(
                self.text_proj,
                checkpoint["text_proj"],
                strict=strict,
                name="text_proj",
            )

        if (
            self.config.training.use_image_proj
            and "image_proj" in checkpoint
            and checkpoint["image_proj"] is not None
        ):
            self._load_state_dict_clean(
                self.image_proj,
                checkpoint["image_proj"],
                strict=strict,
                name="image_proj",
            )

        if resume:
            if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.config.training.use_openclip_optimizer_scheduler == False:
                if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
                    self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.epoch = checkpoint.get("epoch", 0)
            self.step = checkpoint.get("step", 0)
        else:
            self.epoch = 0
            self.step = 0

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

    def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

    def train_one_epoch(self):
        self.model.train()
        if self.config.training.use_text_proj: #False
            self.text_proj.train()
        if self.config.training.use_image_proj: #False
            self.image_proj.train()

        text_contras_acc_list = []
        img_contras_acc_list = []
        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
            mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

        for data in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            loss = 0
            if not self.config.model.get("use_dense", False):
                pred_feat = self.model(data['xyz'], data['features'], \
                                       device=self.config.device, \
                                       quantization_size=self.config.model.voxel_size)
            else:
                pred_feat = self.model(data['xyz_dense'], data['features_dense'])
            logit_scale = self.logit_scale(None)
            idx = data['has_text_idx']

            text_feat = torch.vstack(data['text_feat']).to(self.config.device)
            img_feat = torch.vstack(data['img_feat']).to(self.config.device)

            if self.config.training.use_mask:
                img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
                mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                mask = torch.logical_or(mask, mask_other).detach()
            else:
                mask = None

            if self.clip_adapter is not None:
                with torch.no_grad():
                    img_feat = self.clip_adapter(img_feat)


            if self.config.training.use_image_proj:
                img_feat = self.image_proj(img_feat)
            img_contras_loss, img_contras_acc = self.contras_loss(pred_feat, img_feat, logit_scale=logit_scale,
                                                                  mask=mask)

            loss += img_contras_loss * self.config.training.lambda_img_contras


            if len(idx) > 0:
                if self.config.training.use_text_proj:
                    text_feat = self.text_proj(text_feat)
                text_contras_loss, text_contras_acc = self.contras_loss(pred_feat[idx], text_feat,
                                                                        logit_scale=logit_scale, mask=mask)

                loss += text_contras_loss * self.config.training.lambda_text_contras



            text_contras_acc_list.append(text_contras_acc.item())
            img_contras_acc_list.append(img_contras_acc.item())
            loss.backward()
            self.optimizer.step()
            if self.config.training.use_openclip_optimizer_scheduler:
                self.scheduler(self.step)
            else:
                self.scheduler.step()

        if self.rank == 0:
            logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}' \
                         .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
                                 np.mean(img_contras_acc_list)))

    def save_model(self, name):
        torch.save({
            "state_dict": self.model.state_dict(),
            "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
            "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
            "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.use_openclip_optimizer_scheduler == False else None,
            "epoch": self.epoch,
            "step": self.step,
            "best_img_contras_acc": self.best_img_contras_acc,
            "best_text_contras_acc": self.best_text_contras_acc,
            "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
            "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
            "best_lvis_acc": self.best_lvis_acc,
        }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct

    # -------------------------------------------------------------------------
    # OpenShape compression baselines: truncated and PCA.
    # These methods are eval-only: the OpenShape encoder is frozen.
    # -------------------------------------------------------------------------

    def _get_eval_dims(self):
        """Dimensions used by compression baselines."""
        eval_cfg = self._cfg_get(self.config, "eval", None)

        dims = self._cfg_get(eval_cfg, "pca_dims", None)
        if dims is None:
            dims = self._cfg_get(eval_cfg, "truncated_dims", None)
        if dims is None:
            dims = self._cfg_get(self._cfg_get(self.config, "mrl", None), "nesting_dims", None)
        if dims is None:
            dims = self._cfg_get(self._cfg_get(self.config, "model", None), "nesting_list", None)
        if dims is None:
            dims = [10, 20, 40, 80, 160, 320, 640, 1280]

        return [int(d) for d in dims]

    def _forward_pointcloud(self, data):
        """Forward pass for the 3D encoder, shared by normal/truncated/PCA evaluation."""
        if not self.config.model.get("use_dense", False):
            return self.model(
                data["xyz"],
                data["features"],
                device=self.config.device,
                quantization_size=self.config.model.voxel_size,
            )
        return self.model(data["xyz_dense"], data["features_dense"])

    def _merge_features_dist(self, feat_list):
        """
        Merge variable-length feature tensors across DDP ranks.
        Returns the merged tensor only on rank 0; other ranks receive None.
        """
        feat_local = torch.cat(feat_list, dim=0).contiguous()

        if (not dist.is_available()) or (not dist.is_initialized()) or self.config.ngpu <= 1:
            return feat_local

        world_size = dist.get_world_size()
        device = feat_local.device

        local_n = torch.tensor([feat_local.shape[0]], device=device, dtype=torch.long)
        sizes_t = [torch.zeros_like(local_n) for _ in range(world_size)]
        dist.all_gather(sizes_t, local_n)

        sizes = [int(x.item()) for x in sizes_t]
        max_n = max(sizes)
        feat_dim = feat_local.shape[1]

        if feat_local.shape[0] < max_n:
            pad_n = max_n - feat_local.shape[0]
            feat_pad = torch.zeros(pad_n, feat_dim, device=device, dtype=feat_local.dtype)
            feat_local = torch.cat([feat_local, feat_pad], dim=0)

        gathered_feats = [torch.zeros_like(feat_local) for _ in range(world_size)]
        dist.all_gather(gathered_feats, feat_local)

        if self.rank == 0:
            return torch.cat(
                [gathered_feats[r][:sizes[r]] for r in range(world_size)],
                dim=0,
            )

        return None

    def _merge_features_labels_dist(self, feat_list, label_list):
        """
        Merge variable-length feature/label tensors across DDP ranks.
        Returns merged tensors only on rank 0; other ranks receive (None, None).
        """
        feat_local = torch.cat(feat_list, dim=0).contiguous()
        labels_local = torch.cat(label_list, dim=0).contiguous()

        if (not dist.is_available()) or (not dist.is_initialized()) or self.config.ngpu <= 1:
            return feat_local, labels_local

        world_size = dist.get_world_size()
        device = feat_local.device

        local_n = torch.tensor([feat_local.shape[0]], device=device, dtype=torch.long)
        sizes_t = [torch.zeros_like(local_n) for _ in range(world_size)]
        dist.all_gather(sizes_t, local_n)

        sizes = [int(x.item()) for x in sizes_t]
        max_n = max(sizes)
        feat_dim = feat_local.shape[1]

        if feat_local.shape[0] < max_n:
            pad_n = max_n - feat_local.shape[0]
            feat_pad = torch.zeros(pad_n, feat_dim, device=device, dtype=feat_local.dtype)
            label_pad = torch.zeros(pad_n, device=device, dtype=labels_local.dtype)
            feat_local = torch.cat([feat_local, feat_pad], dim=0)
            labels_local = torch.cat([labels_local, label_pad], dim=0)

        gathered_feats = [torch.zeros_like(feat_local) for _ in range(world_size)]
        gathered_labels = [torch.zeros_like(labels_local) for _ in range(world_size)]

        dist.all_gather(gathered_feats, feat_local)
        dist.all_gather(gathered_labels, labels_local)

        if self.rank == 0:
            feat_all = torch.cat(
                [gathered_feats[r][:sizes[r]] for r in range(world_size)],
                dim=0,
            )
            labels_all = torch.cat(
                [gathered_labels[r][:sizes[r]] for r in range(world_size)],
                dim=0,
            )
            return feat_all, labels_all

        return None, None

    def _compute_zero_shot_stats(self, logits_all, labels_all, num_classes):
        """Compute overall accuracy, mean class accuracy, and top-k accuracy."""
        device = logits_all.device
        labels_all = labels_all.to(device)

        per_cat_correct = torch.zeros(num_classes, device=device)
        per_cat_count = torch.zeros(num_classes, device=device)

        topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

        for i in torch.unique(labels_all):
            idx = labels_all == i
            if idx.sum() > 0:
                per_cat_correct[i] = (
                    logits_all[idx].argmax(dim=1) == labels_all[idx]
                ).float().sum()
                per_cat_count[i] = idx.sum()

        valid = per_cat_count > 0
        overall_acc = per_cat_correct.sum() / per_cat_count.sum()
        class_acc = (per_cat_correct[valid] / per_cat_count[valid]).mean()

        return {
            "overall_acc": overall_acc,
            "class_acc": class_acc,
            "top1": topk_acc[0],
            "top3": topk_acc[1],
            "top5": topk_acc[2],
        }

    def test_zero_shot_truncated(self, loader, dataset_name, num_classes):
        """
        OpenShape truncated baseline.

        Protocol:
        1) Extract full OpenShape 3D features, usually 1280-D.
        2) Merge features across DDP ranks once.
        3) For each d, evaluate normalize(feat[:, :d]) @ normalize(text[:, :d]).T.
        """
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()

        clip_text_feat = torch.from_numpy(loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

        feat_list = []
        labels_list = []

        with torch.no_grad():
            for data in tqdm(loader, desc="Extracting {} features".format(dataset_name)):
                pred_feat = self._forward_pointcloud(data)
                labels = data["category"].to(self.config.device)
                feat_list.append(pred_feat.detach())
                labels_list.append(labels.detach())

        feat_all, labels_all = self._merge_features_labels_dist(feat_list, labels_list)

        if self.rank != 0:
            return None

        clip_text_feat = clip_text_feat.to(feat_all.device)
        max_dim = min(feat_all.shape[1], clip_text_feat.shape[1])
        dims = [d for d in self._get_eval_dims() if d <= max_dim]

        if len(dims) == 0:
            raise ValueError("No valid eval dimensions. max_dim={} dims={}".format(max_dim, self._get_eval_dims()))

        logging.info("========== OpenShape truncated: {} ==========".format(dataset_name))
        logging.info("Feature dim: shape={} text={} | eval dims={}".format(
            tuple(feat_all.shape), tuple(clip_text_feat.shape), dims
        ))

        results = {}
        for d in dims:
            shape_d = F.normalize(feat_all[:, :d], dim=1)
            text_d = F.normalize(clip_text_feat[:, :d], dim=1)
            logits_all = shape_d @ text_d.T

            stats = self._compute_zero_shot_stats(logits_all, labels_all, num_classes)
            results[d] = {k: float(v.item()) for k, v in stats.items()}

            logging.info(
                "OpenShape truncated | {} | D={}: overall_acc: {:.4f} class_acc: {:.4f} "
                "top1_acc: {:.4f} top3_acc: {:.4f} top5_acc: {:.4f}".format(
                    dataset_name,
                    d,
                    results[d]["overall_acc"],
                    results[d]["class_acc"],
                    results[d]["top1"],
                    results[d]["top3"],
                    results[d]["top5"],
                )
            )

        return results

    def test_modelnet40_truncated(self):
        return self.test_zero_shot_truncated(self.modelnet40_loader, "ModelNet40", 40)

    def test_objaverse_lvis_truncated(self):
        return self.test_zero_shot_truncated(self.objaverse_lvis_loader, "ObjaverseLVIS", 1156)

    def test_scanobjectnn_truncated(self):
        return self.test_zero_shot_truncated(self.scanobjectnn_loader, "ScanObjectNN", 15)

    def fit_pca_from_train_loader(self):
        """
        Fit a linear PCA baseline on frozen 1280-D OpenShape training embeddings.

        This uses sklearn.decomposition.PCA to fit a *linear* projection matrix.
        The projection is fitted only on rank 0 after merging train features.
        Evaluation also runs on rank 0 after merging features.
        """
        self.model.eval()

        eval_cfg = self._cfg_get(self.config, "eval", None)
        max_samples = self._cfg_get(eval_cfg, "pca_fit_samples", 20000)
        seed = int(self._cfg_get(eval_cfg, "pca_seed", 0))
        pca_solver = self._cfg_get(eval_cfg, "pca_solver", "randomized")
        pca_whiten = bool(self._cfg_get(eval_cfg, "pca_whiten", False))
        pca_niter = int(self._cfg_get(eval_cfg, "pca_niter", 2))

        feat_list = []
        with torch.no_grad():
            for data in tqdm(self.train_loader, desc="Extracting train features for Linear PCA"):
                pred_feat = self._forward_pointcloud(data)
                feat_list.append(pred_feat.detach())

        feat_all = self._merge_features_dist(feat_list)

        if self.rank != 0:
            return None

        # sklearn PCA expects a NumPy array on CPU.
        feat_all = feat_all.float().cpu().numpy()

        if max_samples is not None and int(max_samples) > 0 and feat_all.shape[0] > int(max_samples):
            rng = np.random.default_rng(seed)
            idx = rng.choice(feat_all.shape[0], size=int(max_samples), replace=False)
            feat_all = feat_all[idx]

        dims = self._get_eval_dims()
        max_dim = min(max(dims), feat_all.shape[0], feat_all.shape[1])

        logging.info(
            "Fitting OpenShape Linear PCA with n_components={} using {} samples | "
            "solver={} whiten={}.".format(
                max_dim,
                feat_all.shape[0],
                pca_solver,
                pca_whiten,
            )
        )

        # Linear PCA baseline. This is the standard post-hoc compression baseline.
        # whiten=False is recommended for the main paper table.
        pca = PCA(
            n_components=max_dim,
            svd_solver=pca_solver,
            whiten=pca_whiten,
            random_state=seed,
            iterated_power=pca_niter,
        )
        pca.fit(feat_all)

        # Store the fitted linear projection in torch tensors so _apply_pca can run
        # without keeping the sklearn object alive.
        self.pca_mean = torch.from_numpy(pca.mean_).float().contiguous()
        self.pca_components = torch.from_numpy(pca.components_).float().contiguous()
        self.pca_explained_variance = torch.from_numpy(pca.explained_variance_).float().contiguous()
        self.pca_explained_variance_ratio = torch.from_numpy(pca.explained_variance_ratio_).float().contiguous()
        self.pca_cumulative_variance_ratio = torch.cumsum(
            self.pca_explained_variance_ratio,
            dim=0,
        ).contiguous()

        # Whitening changes the transform performed by sklearn. If you set whiten=True,
        # apply the same scaling manually in _apply_pca.
        self.pca_whiten = pca_whiten
        if pca_whiten:
            # sklearn PCA whitening scales components by sqrt(n_samples - 1) / singular_values.
            self.pca_singular_values = torch.from_numpy(pca.singular_values_).float().contiguous()
            self.pca_fit_n_samples = int(pca.n_samples_)
        else:
            self.pca_singular_values = None
            self.pca_fit_n_samples = int(pca.n_samples_)

        for dim in dims:
            if dim <= self.pca_cumulative_variance_ratio.shape[0]:
                var_pct = self._get_pca_explained_variance(dim)
                logging.info(
                    "OpenShape Linear PCA | D={}: explained_variance_preserved: {:.2f}%".format(
                        dim,
                        var_pct,
                    )
                )

        cache_path = self._cfg_get(eval_cfg, "pca_cache_path", None)
        if cache_path is None:
            cache_path = os.path.join(self.config.ckpt_dir, "openshape_linear_pca_fit.pt")

        torch.save(
            {
                "mean": self.pca_mean,
                "components": self.pca_components,
                "explained_variance": self.pca_explained_variance,
                "explained_variance_ratio": self.pca_explained_variance_ratio,
                "cumulative_variance_ratio": self.pca_cumulative_variance_ratio,
                "whiten": self.pca_whiten,
                "singular_values": self.pca_singular_values,
                "fit_n_samples": self.pca_fit_n_samples,
                "dims": dims,
                "fit_samples": feat_all.shape[0],
                "solver": pca_solver,
                "sklearn_pca": True,
                "pca_type": "linear",
            },
            cache_path,
        )
        logging.info("Saved Linear Linear PCA cache to {}".format(cache_path))
        logging.info("Finished fitting Linear PCA.")
        return cache_path

    def load_pca(self, path=None):
        """Load a previously fitted Linear PCA cache."""
        eval_cfg = self._cfg_get(self.config, "eval", None)
        if path is None:
            path = self._cfg_get(eval_cfg, "pca_cache_path", None)
        if path is None:
            path = os.path.join(self.config.ckpt_dir, "openshape_linear_pca_fit.pt")

        pca = torch.load(path, map_location="cpu")
        self.pca_mean = pca["mean"].float().contiguous()
        self.pca_components = pca["components"].float().contiguous()
        self.pca_whiten = bool(pca.get("whiten", False))
        self.pca_singular_values = pca.get("singular_values", None)
        if self.pca_singular_values is not None:
            self.pca_singular_values = self.pca_singular_values.float().contiguous()
        self.pca_fit_n_samples = int(pca.get("fit_n_samples", pca.get("fit_samples", 0)))

        if "explained_variance" in pca:
            self.pca_explained_variance = pca["explained_variance"].float().contiguous()
        if "explained_variance_ratio" in pca:
            self.pca_explained_variance_ratio = pca["explained_variance_ratio"].float().contiguous()
        if "cumulative_variance_ratio" in pca:
            self.pca_cumulative_variance_ratio = pca["cumulative_variance_ratio"].float().contiguous()
        elif hasattr(self, "pca_explained_variance_ratio"):
            self.pca_cumulative_variance_ratio = torch.cumsum(
                self.pca_explained_variance_ratio,
                dim=0,
            ).contiguous()

        if hasattr(self, "pca_cumulative_variance_ratio"):
            for dim in self._get_eval_dims():
                if dim <= self.pca_cumulative_variance_ratio.shape[0]:
                    logging.info(
                        "OpenShape Linear PCA | D={}: explained_variance_preserved: {:.2f}%".format(
                            dim,
                            self._get_pca_explained_variance(dim),
                        )
                    )

        logging.info("Loaded Linear PCA cache from {}".format(path))

    def _get_pca_explained_variance(self, dim):
        """Return cumulative PCA explained variance up to dim, in percent."""
        if not hasattr(self, "pca_cumulative_variance_ratio"):
            return None
        if dim <= 0 or dim > self.pca_cumulative_variance_ratio.shape[0]:
            return None
        return float(self.pca_cumulative_variance_ratio[dim - 1].item() * 100.0)

    def _apply_pca(self, feat, dim):
        """Apply fitted PCA to a feature tensor and return an [N, dim] CPU tensor."""
        if not hasattr(self, "pca_mean") or not hasattr(self, "pca_components"):
            raise RuntimeError("PCA is not fitted. Call fit_pca_from_train_loader() or load_pca() first.")

        if dim > self.pca_components.shape[0]:
            raise ValueError("Requested dim={} but PCA has only {} components.".format(
                dim, self.pca_components.shape[0]
            ))

        feat = feat.float().cpu()
        mean = self.pca_mean.float().cpu()
        components = self.pca_components[:dim].float().cpu()
        projected = (feat - mean) @ components.T

        # Match sklearn PCA.transform when whiten=True.
        # Main baseline should use whiten=False, but this keeps the code correct
        # for optional whitening ablations.
        if getattr(self, "pca_whiten", False):
            if self.pca_singular_values is None or self.pca_fit_n_samples <= 1:
                raise RuntimeError("PCA whitening requested, but singular_values/fit_n_samples are missing from cache.")
            scale = (float(self.pca_fit_n_samples - 1) ** 0.5) / self.pca_singular_values[:dim].float().cpu().clamp_min(1e-12)
            projected = projected * scale

        return projected

    def test_zero_shot_pca(self, loader, dataset_name, num_classes):
        """
        OpenShape + PCA baseline.

        The same PCA projection is applied to shape and text embeddings, then
        features are L2-normalized and evaluated with cosine similarity.
        """
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()

        clip_text_feat = torch.from_numpy(loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

        feat_list = []
        labels_list = []

        with torch.no_grad():
            for data in tqdm(loader, desc="Extracting {} features for PCA eval".format(dataset_name)):
                pred_feat = self._forward_pointcloud(data)
                labels = data["category"].to(self.config.device)
                feat_list.append(pred_feat.detach())
                labels_list.append(labels.detach())

        feat_all, labels_all = self._merge_features_labels_dist(feat_list, labels_list)

        if self.rank != 0:
            return None

        dims = [d for d in self._get_eval_dims() if d <= self.pca_components.shape[0]]

        if len(dims) == 0:
            raise ValueError("No valid PCA dimensions. PCA components={} dims={}".format(
                self.pca_components.shape[0], self._get_eval_dims()
            ))

        labels_all = labels_all.cpu()
        logging.info("========== OpenShape Linear PCA: {} ==========".format(dataset_name))
        logging.info("Feature dim: shape={} text={} | eval dims={}".format(
            tuple(feat_all.shape), tuple(clip_text_feat.shape), dims
        ))

        results = {}
        for d in dims:
            shape_d = self._apply_pca(feat_all, d)
            text_d = self._apply_pca(clip_text_feat, d)

            shape_d = F.normalize(shape_d, dim=1)
            text_d = F.normalize(text_d, dim=1)
            logits_all = shape_d @ text_d.T

            stats = self._compute_zero_shot_stats(logits_all, labels_all, num_classes)
            results[d] = {k: float(v.item()) for k, v in stats.items()}

            var_preserved = self._get_pca_explained_variance(d)
            results[d]["explained_variance_preserved"] = var_preserved

            if var_preserved is not None:
                logging.info(
                    "OpenShape Linear PCA | {} | D={}: explained_variance_preserved: {:.2f}% "
                    "overall_acc: {:.4f} class_acc: {:.4f} "
                    "top1_acc: {:.4f} top3_acc: {:.4f} top5_acc: {:.4f}".format(
                        dataset_name,
                        d,
                        var_preserved,
                        results[d]["overall_acc"],
                        results[d]["class_acc"],
                        results[d]["top1"],
                        results[d]["top3"],
                        results[d]["top5"],
                    )
                )
            else:
                logging.info(
                    "OpenShape Linear PCA | {} | D={}: overall_acc: {:.4f} class_acc: {:.4f} "
                    "top1_acc: {:.4f} top3_acc: {:.4f} top5_acc: {:.4f}".format(
                        dataset_name,
                        d,
                        results[d]["overall_acc"],
                        results[d]["class_acc"],
                        results[d]["top1"],
                        results[d]["top3"],
                        results[d]["top5"],
                    )
                )

        return results

    def test_modelnet40_pca(self):
        return self.test_zero_shot_pca(self.modelnet40_loader, "ModelNet40", 40)

    def test_objaverse_lvis_pca(self):
        return self.test_zero_shot_pca(self.objaverse_lvis_loader, "ObjaverseLVIS", 1156)

    def test_scanobjectnn_pca(self):
        return self.test_zero_shot_pca(self.scanobjectnn_loader, "ScanObjectNN", 15)

    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()
            if epoch > self.config.training.test_epoch:
                self.test_objaverse_lvis()
                # self.test_modelnet40()
                self.test_objaverse_lvis()
            if self.rank == 0:
                self.save_model('latest')
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))


    def test_modelnet40(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).cuda()
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(40).cuda()
        per_cat_count = torch.zeros(40).cuda()
        category2idx = self.modelnet40_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.modelnet40_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "modelnet40_dir"),
                                                    logits_all, labels_all)

        if self.rank == 0:

            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            for i in range(40):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count
            # for i in range(40):
            #    print(idx2category[i], per_cat_acc[i])

            if overall_acc > self.best_modelnet40_overall_acc:
                self.best_modelnet40_overall_acc = overall_acc
                # self.save_model('best_modelnet40_overall')
            if per_cat_acc.mean() > self.best_modelnet40_class_acc:
                self.best_modelnet40_class_acc = per_cat_acc.mean()
                # self.save_model('best_modelnet40_class')

            logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
                                                                                             self.best_modelnet40_overall_acc,
                                                                                             per_cat_acc.mean(),
                                                                                             self.best_modelnet40_class_acc))
            logging.info(
                'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                    topk_acc[1].item(),
                                                                                    topk_acc[2].item()))
            # wandb.log({"test/epoch": self.epoch,
            #            "test/step": self.step,
            #            "test/ModelNet40_overall_acc": overall_acc,
            #            "test/ModelNet40_class_acc": per_cat_acc.mean(),
            #            "test/top3_acc": topk_acc[1],
            #            "test/top5_acc": topk_acc[2], })

    def test_objaverse_lvis(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).cuda()
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(1156).cuda()
        per_cat_count = torch.zeros(1156).cuda()
        category2idx = self.objaverse_lvis_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.objaverse_lvis_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "objaverse_dir"),
                                                    logits_all, labels_all)

        if self.rank == 0:
            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            # calculate per class accuracy
            for i in torch.unique(labels_all):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            if overall_acc > self.best_lvis_acc:
                self.best_lvis_acc = overall_acc
                # self.save_model('best_lvis')

            logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
            logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                                topk_acc[1].item(),
                                                                                                topk_acc[2].item()))
            # wandb.log({"test_lvis/epoch": self.epoch,
            #            "test_lvis/step": self.step,
            #            "test_lvis/overall_acc": overall_acc,
            #            "test_lvis/class_acc": per_cat_acc.mean(),
            #            "test_lvis/top3_acc": topk_acc[1],
            #            "test_lvis/top5_acc": topk_acc[2], })

    def test_scanobjectnn(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count = torch.zeros(15).to(self.config.device)
        category2idx = self.scanobjectnn_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.scanobjectnn_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scanobjectnn_dir"),
                                                    logits_all, labels_all)

        if self.rank == 0:

            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            # calculate per class accuracy
            for i in range(15):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
            logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                               topk_acc[1].item(),
                                                                                               topk_acc[2].item()))
            # wandb.log({"test_scanobjectnn/epoch": self.epoch,
            #            "test_scanobjectnn/step": self.step,
            #            "test_scanobjectnn/overall_acc": overall_acc,
            #            "test_scanobjectnn/class_acc": per_cat_acc.mean(),
            #            "test_scanobjectnn/top3_acc": topk_acc[1],
            #            "test_scanobjectnn/top5_acc": topk_acc[2], })


    def test_scannet(self, scannet_loader):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(scannet_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(19).to(self.config.device)
        per_cat_count = torch.zeros(19).to(self.config.device)
        category2idx = scannet_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_all = []
        labels_all = []
        with torch.no_grad():
            for data in tqdm(scannet_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scannet_dir"),
                                                    logits_all, labels_all)
        if self.rank == 0:

            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            # calculate per class accuracy
            for i in range(19):
                idx = (logits_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count


            logging.info('Test Scannet: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
            logging.info('Test Scannet: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                           topk_acc[1].item(),
                                                                                               topk_acc[2].item()))

# Backward-compatible alias used by your main/test script.
# TrainerOpenShape = Trainer

# import logging
# import os

# import numpy as np
# import torch
# import torch.distributed.nn
# import torch.nn.functional as F
# import wandb
# import torch.distributed as dist
# from trainers.trainer_utils import merge_results_dist
# from tqdm import tqdm
# from trainers.trainer_utils import merge_two_branch_results_dist


# class TrainerOpenShape(object):
#     def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, train_loader, \
#                  modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None, clip_adapter=None):
#         self.rank = rank
#         self.config = config
#         self.model = model
#         self.logit_scale = logit_scale
#         self.image_proj = image_proj
#         self.text_proj = text_proj
#         self.optimizer = optimizer
#         self.scheduler = scheduler
#         self.train_loader = train_loader
#         self.modelnet40_loader = modelnet40_loader
#         self.objaverse_lvis_loader = objaverse_lvis_loader
#         self.scanobjectnn_loader = scanobjectnn_loader
#         self.epoch = 0
#         self.step = 0
#         self.clip_adapter = clip_adapter
#         self.best_img_contras_acc = 0
#         self.best_text_contras_acc = 0
#         self.best_modelnet40_overall_acc = 0
#         self.best_modelnet40_class_acc = 0
#         self.best_lvis_acc = 0
#         self.config.ngpu = dist.get_world_size()

#     def _cfg_get(self, obj, key, default=None):
#         """Read config values from dict/OmegaConf-like objects safely."""
#         if obj is None:
#             return default
#         if isinstance(obj, dict):
#             return obj.get(key, default)
#         try:
#             return obj.get(key, default)
#         except Exception:
#             return getattr(obj, key, default)

#     def _get_module(self, module):
#         """Return the wrapped module when using DataParallel/DDP."""
#         return module.module if hasattr(module, "module") else module

#     def _clean_ddp_state_dict(self, state_dict):
#         """Remove the 'module.' prefix from checkpoints saved with DDP."""
#         if state_dict is None:
#             return None
#         return {
#             k[len("module."):] if k.startswith("module.") else k: v
#             for k, v in state_dict.items()
#         }

#     def _load_state_dict_clean(self, module, state_dict, strict=True, name="module"):
#         """Load a state_dict robustly, handling DDP/non-DDP checkpoints."""
#         state_dict = self._clean_ddp_state_dict(state_dict)
#         msg = self._get_module(module).load_state_dict(state_dict, strict=strict)
#         logging.info("Loaded {} state_dict: {}".format(name, msg))
#         return msg

#     def load_from_checkpoint(self, path, resume=True, strict=True):
#         """
#         Load OpenShape checkpoint.

#         Use resume=False for evaluation-only compression baselines, e.g.
#         OpenShape truncated and OpenShape PCA. This avoids requiring optimizer
#         and scheduler states when you only want to test a frozen checkpoint.
#         """
#         checkpoint = torch.load(path, map_location="cpu")

#         self._load_state_dict_clean(
#             self.model,
#             checkpoint["state_dict"],
#             strict=strict,
#             name="model",
#         )

#         if "logit_scale" in checkpoint and checkpoint["logit_scale"] is not None:
#             self._load_state_dict_clean(
#                 self.logit_scale,
#                 checkpoint["logit_scale"],
#                 strict=strict,
#                 name="logit_scale",
#             )

#         if (
#             self.config.training.use_text_proj
#             and "text_proj" in checkpoint
#             and checkpoint["text_proj"] is not None
#         ):
#             self._load_state_dict_clean(
#                 self.text_proj,
#                 checkpoint["text_proj"],
#                 strict=strict,
#                 name="text_proj",
#             )

#         if (
#             self.config.training.use_image_proj
#             and "image_proj" in checkpoint
#             and checkpoint["image_proj"] is not None
#         ):
#             self._load_state_dict_clean(
#                 self.image_proj,
#                 checkpoint["image_proj"],
#                 strict=strict,
#                 name="image_proj",
#             )

#         if resume:
#             if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
#                 self.optimizer.load_state_dict(checkpoint["optimizer"])
#             if self.config.training.use_openclip_optimizer_scheduler == False:
#                 if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
#                     self.scheduler.load_state_dict(checkpoint["scheduler"])
#             self.epoch = checkpoint.get("epoch", 0)
#             self.step = checkpoint.get("step", 0)
#         else:
#             self.epoch = 0
#             self.step = 0

#         logging.info("Loaded checkpoint from {}".format(path))
#         logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

#     def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
#         if self.config.ngpu > 1:
#             feat1 = F.normalize(feat1, dim=1)
#             feat2 = F.normalize(feat2, dim=1)
#             all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
#             all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
#             logits = logit_scale * all_feat1 @ all_feat2.T
#         else:
#             logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
#         if mask is not None:
#             logits = logits * mask
#         labels = torch.arange(logits.shape[0]).to(self.config.device)
#         accuracy = (logits.argmax(dim=1) == labels).float().mean()
#         loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
#         return loss, accuracy

#     def train_one_epoch(self):
#         self.model.train()
#         if self.config.training.use_text_proj: #False
#             self.text_proj.train()
#         if self.config.training.use_image_proj: #False
#             self.image_proj.train()

#         text_contras_acc_list = []
#         img_contras_acc_list = []
#         if self.config.training.use_mask:
#             k = self.config.dataset.negative_sample_num
#             s = self.config.dataset.train_batch_size
#             mask1 = np.eye(k * s).astype(np.bool)
#             mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
#             mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

#         for data in tqdm(self.train_loader):
#             self.step += 1
#             self.optimizer.zero_grad()
#             loss = 0
#             if not self.config.model.get("use_dense", False):
#                 pred_feat = self.model(data['xyz'], data['features'], \
#                                        device=self.config.device, \
#                                        quantization_size=self.config.model.voxel_size)
#             else:
#                 pred_feat = self.model(data['xyz_dense'], data['features_dense'])
#             logit_scale = self.logit_scale(None)
#             idx = data['has_text_idx']

#             text_feat = torch.vstack(data['text_feat']).to(self.config.device)
#             img_feat = torch.vstack(data['img_feat']).to(self.config.device)

#             if self.config.training.use_mask:
#                 img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
#                 mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
#                 mask = torch.logical_or(mask, mask_other).detach()
#             else:
#                 mask = None

#             if self.clip_adapter is not None:
#                 with torch.no_grad():
#                     img_feat = self.clip_adapter(img_feat)


#             if self.config.training.use_image_proj:
#                 img_feat = self.image_proj(img_feat)
#             img_contras_loss, img_contras_acc = self.contras_loss(pred_feat, img_feat, logit_scale=logit_scale,
#                                                                   mask=mask)

#             loss += img_contras_loss * self.config.training.lambda_img_contras


#             if len(idx) > 0:
#                 if self.config.training.use_text_proj:
#                     text_feat = self.text_proj(text_feat)
#                 text_contras_loss, text_contras_acc = self.contras_loss(pred_feat[idx], text_feat,
#                                                                         logit_scale=logit_scale, mask=mask)

#                 loss += text_contras_loss * self.config.training.lambda_text_contras



#             text_contras_acc_list.append(text_contras_acc.item())
#             img_contras_acc_list.append(img_contras_acc.item())
#             loss.backward()
#             self.optimizer.step()
#             if self.config.training.use_openclip_optimizer_scheduler:
#                 self.scheduler(self.step)
#             else:
#                 self.scheduler.step()

#         if self.rank == 0:
#             logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}' \
#                          .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
#                                  np.mean(img_contras_acc_list)))

#     def save_model(self, name):
#         torch.save({
#             "state_dict": self.model.state_dict(),
#             "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
#             "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
#             "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
#             "optimizer": self.optimizer.state_dict(),
#             "scheduler": self.scheduler.state_dict() if self.config.training.use_openclip_optimizer_scheduler == False else None,
#             "epoch": self.epoch,
#             "step": self.step,
#             "best_img_contras_acc": self.best_img_contras_acc,
#             "best_text_contras_acc": self.best_text_contras_acc,
#             "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
#             "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
#             "best_lvis_acc": self.best_lvis_acc,
#         }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

#     def accuracy(self, output, target, topk=(1,)):
#         """Computes the accuracy over the k top predictions for the specified values of k"""
#         with torch.no_grad():
#             maxk = max(topk)
#             batch_size = target.size(0)

#             _, pred = output.topk(maxk, 1, True, True)
#             pred = pred.t()
#             correct = pred.eq(target.reshape(1, -1).expand_as(pred))

#             res = []
#             for k in topk:
#                 correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#                 res.append(correct_k.mul_(100.0 / batch_size))
#             return res, correct

#     # -------------------------------------------------------------------------
#     # OpenShape compression baselines: truncated and PCA.
#     # These methods are eval-only: the OpenShape encoder is frozen.
#     # -------------------------------------------------------------------------

#     def _get_eval_dims(self):
#         """Dimensions used by compression baselines."""
#         eval_cfg = self._cfg_get(self.config, "eval", None)

#         dims = self._cfg_get(eval_cfg, "pca_dims", None)
#         if dims is None:
#             dims = self._cfg_get(eval_cfg, "truncated_dims", None)
#         if dims is None:
#             dims = self._cfg_get(self._cfg_get(self.config, "mrl", None), "nesting_dims", None)
#         if dims is None:
#             dims = self._cfg_get(self._cfg_get(self.config, "model", None), "nesting_list", None)
#         if dims is None:
#             dims = [10, 20, 40, 80, 160, 320, 640, 1280]

#         return [int(d) for d in dims]

#     def _forward_pointcloud(self, data):
#         """Forward pass for the 3D encoder, shared by normal/truncated/PCA evaluation."""
#         if not self.config.model.get("use_dense", False):
#             return self.model(
#                 data["xyz"],
#                 data["features"],
#                 device=self.config.device,
#                 quantization_size=self.config.model.voxel_size,
#             )
#         return self.model(data["xyz_dense"], data["features_dense"])

#     def _merge_features_dist(self, feat_list):
#         """
#         Merge variable-length feature tensors across DDP ranks.
#         Returns the merged tensor only on rank 0; other ranks receive None.
#         """
#         feat_local = torch.cat(feat_list, dim=0).contiguous()

#         if (not dist.is_available()) or (not dist.is_initialized()) or self.config.ngpu <= 1:
#             return feat_local

#         world_size = dist.get_world_size()
#         device = feat_local.device

#         local_n = torch.tensor([feat_local.shape[0]], device=device, dtype=torch.long)
#         sizes_t = [torch.zeros_like(local_n) for _ in range(world_size)]
#         dist.all_gather(sizes_t, local_n)

#         sizes = [int(x.item()) for x in sizes_t]
#         max_n = max(sizes)
#         feat_dim = feat_local.shape[1]

#         if feat_local.shape[0] < max_n:
#             pad_n = max_n - feat_local.shape[0]
#             feat_pad = torch.zeros(pad_n, feat_dim, device=device, dtype=feat_local.dtype)
#             feat_local = torch.cat([feat_local, feat_pad], dim=0)

#         gathered_feats = [torch.zeros_like(feat_local) for _ in range(world_size)]
#         dist.all_gather(gathered_feats, feat_local)

#         if self.rank == 0:
#             return torch.cat(
#                 [gathered_feats[r][:sizes[r]] for r in range(world_size)],
#                 dim=0,
#             )

#         return None

#     def _merge_features_labels_dist(self, feat_list, label_list):
#         """
#         Merge variable-length feature/label tensors across DDP ranks.
#         Returns merged tensors only on rank 0; other ranks receive (None, None).
#         """
#         feat_local = torch.cat(feat_list, dim=0).contiguous()
#         labels_local = torch.cat(label_list, dim=0).contiguous()

#         if (not dist.is_available()) or (not dist.is_initialized()) or self.config.ngpu <= 1:
#             return feat_local, labels_local

#         world_size = dist.get_world_size()
#         device = feat_local.device

#         local_n = torch.tensor([feat_local.shape[0]], device=device, dtype=torch.long)
#         sizes_t = [torch.zeros_like(local_n) for _ in range(world_size)]
#         dist.all_gather(sizes_t, local_n)

#         sizes = [int(x.item()) for x in sizes_t]
#         max_n = max(sizes)
#         feat_dim = feat_local.shape[1]

#         if feat_local.shape[0] < max_n:
#             pad_n = max_n - feat_local.shape[0]
#             feat_pad = torch.zeros(pad_n, feat_dim, device=device, dtype=feat_local.dtype)
#             label_pad = torch.zeros(pad_n, device=device, dtype=labels_local.dtype)
#             feat_local = torch.cat([feat_local, feat_pad], dim=0)
#             labels_local = torch.cat([labels_local, label_pad], dim=0)

#         gathered_feats = [torch.zeros_like(feat_local) for _ in range(world_size)]
#         gathered_labels = [torch.zeros_like(labels_local) for _ in range(world_size)]

#         dist.all_gather(gathered_feats, feat_local)
#         dist.all_gather(gathered_labels, labels_local)

#         if self.rank == 0:
#             feat_all = torch.cat(
#                 [gathered_feats[r][:sizes[r]] for r in range(world_size)],
#                 dim=0,
#             )
#             labels_all = torch.cat(
#                 [gathered_labels[r][:sizes[r]] for r in range(world_size)],
#                 dim=0,
#             )
#             return feat_all, labels_all

#         return None, None

#     def _compute_zero_shot_stats(self, logits_all, labels_all, num_classes):
#         """Compute overall accuracy, mean class accuracy, and top-k accuracy."""
#         device = logits_all.device
#         labels_all = labels_all.to(device)

#         per_cat_correct = torch.zeros(num_classes, device=device)
#         per_cat_count = torch.zeros(num_classes, device=device)

#         topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

#         for i in torch.unique(labels_all):
#             idx = labels_all == i
#             if idx.sum() > 0:
#                 per_cat_correct[i] = (
#                     logits_all[idx].argmax(dim=1) == labels_all[idx]
#                 ).float().sum()
#                 per_cat_count[i] = idx.sum()

#         valid = per_cat_count > 0
#         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
#         class_acc = (per_cat_correct[valid] / per_cat_count[valid]).mean()

#         return {
#             "overall_acc": overall_acc,
#             "class_acc": class_acc,
#             "top1": topk_acc[0],
#             "top3": topk_acc[1],
#             "top5": topk_acc[2],
#         }

#     def test_zero_shot_truncated(self, loader, dataset_name, num_classes):
#         """
#         OpenShape truncated baseline.

#         Protocol:
#         1) Extract full OpenShape 3D features, usually 1280-D.
#         2) Merge features across DDP ranks once.
#         3) For each d, evaluate normalize(feat[:, :d]) @ normalize(text[:, :d]).T.
#         """
#         self.model.eval()
#         if self.config.training.use_text_proj:
#             self.text_proj.eval()

#         clip_text_feat = torch.from_numpy(loader.dataset.clip_cat_feat).to(self.config.device)
#         if self.config.training.use_text_proj:
#             clip_text_feat = self.text_proj(clip_text_feat)

#         feat_list = []
#         labels_list = []

#         with torch.no_grad():
#             for data in tqdm(loader, desc="Extracting {} features".format(dataset_name)):
#                 pred_feat = self._forward_pointcloud(data)
#                 labels = data["category"].to(self.config.device)
#                 feat_list.append(pred_feat.detach())
#                 labels_list.append(labels.detach())

#         feat_all, labels_all = self._merge_features_labels_dist(feat_list, labels_list)

#         if self.rank != 0:
#             return None

#         clip_text_feat = clip_text_feat.to(feat_all.device)
#         max_dim = min(feat_all.shape[1], clip_text_feat.shape[1])
#         dims = [d for d in self._get_eval_dims() if d <= max_dim]

#         if len(dims) == 0:
#             raise ValueError("No valid eval dimensions. max_dim={} dims={}".format(max_dim, self._get_eval_dims()))

#         logging.info("========== OpenShape truncated: {} ==========".format(dataset_name))
#         logging.info("Feature dim: shape={} text={} | eval dims={}".format(
#             tuple(feat_all.shape), tuple(clip_text_feat.shape), dims
#         ))

#         results = {}
#         for d in dims:
#             shape_d = F.normalize(feat_all[:, :d], dim=1)
#             text_d = F.normalize(clip_text_feat[:, :d], dim=1)
#             logits_all = shape_d @ text_d.T

#             stats = self._compute_zero_shot_stats(logits_all, labels_all, num_classes)
#             results[d] = {k: float(v.item()) for k, v in stats.items()}

#             logging.info(
#                 "OpenShape truncated | {} | D={}: overall_acc: {:.4f} class_acc: {:.4f} "
#                 "top1_acc: {:.4f} top3_acc: {:.4f} top5_acc: {:.4f}".format(
#                     dataset_name,
#                     d,
#                     results[d]["overall_acc"],
#                     results[d]["class_acc"],
#                     results[d]["top1"],
#                     results[d]["top3"],
#                     results[d]["top5"],
#                 )
#             )

#         return results

#     def test_modelnet40_truncated(self):
#         return self.test_zero_shot_truncated(self.modelnet40_loader, "ModelNet40", 40)

#     def test_objaverse_lvis_truncated(self):
#         return self.test_zero_shot_truncated(self.objaverse_lvis_loader, "ObjaverseLVIS", 1156)

#     def test_scanobjectnn_truncated(self):
#         return self.test_zero_shot_truncated(self.scanobjectnn_loader, "ScanObjectNN", 15)

#     def fit_pca_from_train_loader(self):
#         """
#         Fit PCA on frozen 1280-D OpenShape training embeddings.

#         PCA is fitted on rank 0 only. Evaluation also applies the projection on
#         rank 0 after merging features, so other ranks do not need the PCA state.
#         """
#         self.model.eval()

#         eval_cfg = self._cfg_get(self.config, "eval", None)
#         max_samples = self._cfg_get(eval_cfg, "pca_fit_samples", 20000)
#         seed = int(self._cfg_get(eval_cfg, "pca_seed", 0))
#         pca_niter = int(self._cfg_get(eval_cfg, "pca_niter", 2))

#         feat_list = []
#         with torch.no_grad():
#             for data in tqdm(self.train_loader, desc="Extracting train features for PCA"):
#                 pred_feat = self._forward_pointcloud(data)
#                 feat_list.append(pred_feat.detach())

#         feat_all = self._merge_features_dist(feat_list)

#         if self.rank != 0:
#             return None

#         feat_all = feat_all.float().cpu()

#         if max_samples is not None and int(max_samples) > 0 and feat_all.shape[0] > int(max_samples):
#             generator = torch.Generator(device="cpu")
#             generator.manual_seed(seed)
#             perm = torch.randperm(feat_all.shape[0], generator=generator)[: int(max_samples)]
#             feat_all = feat_all[perm]

#         dims = self._get_eval_dims()
#         max_dim = min(max(dims), feat_all.shape[0], feat_all.shape[1])

#         logging.info(
#             "Fitting OpenShape PCA with n_components={} using {} samples.".format(
#                 max_dim, feat_all.shape[0]
#             )
#         )

#         mean = feat_all.mean(dim=0, keepdim=True)
#         centered = feat_all - mean

#         try:
#             # V has shape [in_dim, max_dim]. Components are V.T.
#             # S contains singular values for the selected principal components.
#             _, singular_values, v = torch.pca_lowrank(centered, q=max_dim, center=False, niter=pca_niter)
#             components = v[:, :max_dim].T.contiguous()
#             singular_values = singular_values[:max_dim].contiguous()
#         except Exception as exc:
#             logging.warning("torch.pca_lowrank failed ({}). Falling back to torch.linalg.svd.".format(exc))
#             _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
#             components = vh[:max_dim].contiguous()
#             singular_values = singular_values[:max_dim].contiguous()

#         self.pca_mean = mean.squeeze(0).contiguous()
#         self.pca_components = components.contiguous()

#         # PCA variance accounting.
#         # explained_variance[d] = variance explained by component d.
#         # explained_variance_ratio[d] = fraction of total input variance explained by component d.
#         # cumulative_variance_ratio[d] = fraction preserved by the first d+1 components.
#         denom = max(feat_all.shape[0] - 1, 1)
#         total_variance = centered.pow(2).sum() / denom
#         explained_variance = singular_values.pow(2) / denom
#         explained_variance_ratio = explained_variance / total_variance.clamp_min(1e-12)

#         self.pca_explained_variance = explained_variance.float().cpu().contiguous()
#         self.pca_explained_variance_ratio = explained_variance_ratio.float().cpu().contiguous()
#         self.pca_cumulative_variance_ratio = torch.cumsum(
#             self.pca_explained_variance_ratio,
#             dim=0,
#         ).contiguous()

#         for dim in dims:
#             if dim <= self.pca_cumulative_variance_ratio.shape[0]:
#                 var_pct = self._get_pca_explained_variance(dim)
#                 logging.info(
#                     "OpenShape PCA | D={}: explained_variance_preserved: {:.2f}%".format(
#                         dim,
#                         var_pct,
#                     )
#                 )

#         cache_path = self._cfg_get(eval_cfg, "pca_cache_path", None)
#         if cache_path is None:
#             cache_path = os.path.join(self.config.ckpt_dir, "openshape_pca_fit.pt")

#         torch.save(
#             {
#                 "mean": self.pca_mean,
#                 "components": self.pca_components,
#                 "explained_variance": self.pca_explained_variance,
#                 "explained_variance_ratio": self.pca_explained_variance_ratio,
#                 "cumulative_variance_ratio": self.pca_cumulative_variance_ratio,
#                 "dims": dims,
#                 "fit_samples": feat_all.shape[0],
#             },
#             cache_path,
#         )
#         logging.info("Saved PCA cache to {}".format(cache_path))
#         logging.info("Finished fitting PCA.")
#         return cache_path

#     def load_pca(self, path=None):
#         """Load a previously fitted PCA cache."""
#         eval_cfg = self._cfg_get(self.config, "eval", None)
#         if path is None:
#             path = self._cfg_get(eval_cfg, "pca_cache_path", None)
#         if path is None:
#             path = os.path.join(self.config.ckpt_dir, "openshape_pca_fit.pt")

#         pca = torch.load(path, map_location="cpu")
#         self.pca_mean = pca["mean"].float().contiguous()
#         self.pca_components = pca["components"].float().contiguous()

#         if "explained_variance" in pca:
#             self.pca_explained_variance = pca["explained_variance"].float().contiguous()
#         if "explained_variance_ratio" in pca:
#             self.pca_explained_variance_ratio = pca["explained_variance_ratio"].float().contiguous()
#         if "cumulative_variance_ratio" in pca:
#             self.pca_cumulative_variance_ratio = pca["cumulative_variance_ratio"].float().contiguous()
#         elif hasattr(self, "pca_explained_variance_ratio"):
#             self.pca_cumulative_variance_ratio = torch.cumsum(
#                 self.pca_explained_variance_ratio,
#                 dim=0,
#             ).contiguous()

#         if hasattr(self, "pca_cumulative_variance_ratio"):
#             for dim in self._get_eval_dims():
#                 if dim <= self.pca_cumulative_variance_ratio.shape[0]:
#                     logging.info(
#                         "OpenShape PCA | D={}: explained_variance_preserved: {:.2f}%".format(
#                             dim,
#                             self._get_pca_explained_variance(dim),
#                         )
#                     )

#         logging.info("Loaded PCA cache from {}".format(path))

#     def _get_pca_explained_variance(self, dim):
#         """Return cumulative PCA explained variance up to dim, in percent."""
#         if not hasattr(self, "pca_cumulative_variance_ratio"):
#             return None
#         if dim <= 0 or dim > self.pca_cumulative_variance_ratio.shape[0]:
#             return None
#         return float(self.pca_cumulative_variance_ratio[dim - 1].item() * 100.0)

#     def _apply_pca(self, feat, dim):
#         """Apply fitted PCA to a feature tensor and return an [N, dim] CPU tensor."""
#         if not hasattr(self, "pca_mean") or not hasattr(self, "pca_components"):
#             raise RuntimeError("PCA is not fitted. Call fit_pca_from_train_loader() or load_pca() first.")

#         if dim > self.pca_components.shape[0]:
#             raise ValueError("Requested dim={} but PCA has only {} components.".format(
#                 dim, self.pca_components.shape[0]
#             ))

#         feat = feat.float().cpu()
#         mean = self.pca_mean.float().cpu()
#         components = self.pca_components[:dim].float().cpu()
#         return (feat - mean) @ components.T

#     def test_zero_shot_pca(self, loader, dataset_name, num_classes):
#         """
#         OpenShape + PCA baseline.

#         The same PCA projection is applied to shape and text embeddings, then
#         features are L2-normalized and evaluated with cosine similarity.
#         """
#         self.model.eval()
#         if self.config.training.use_text_proj:
#             self.text_proj.eval()

#         clip_text_feat = torch.from_numpy(loader.dataset.clip_cat_feat).to(self.config.device)
#         if self.config.training.use_text_proj:
#             clip_text_feat = self.text_proj(clip_text_feat)

#         feat_list = []
#         labels_list = []

#         with torch.no_grad():
#             for data in tqdm(loader, desc="Extracting {} features for PCA eval".format(dataset_name)):
#                 pred_feat = self._forward_pointcloud(data)
#                 labels = data["category"].to(self.config.device)
#                 feat_list.append(pred_feat.detach())
#                 labels_list.append(labels.detach())

#         feat_all, labels_all = self._merge_features_labels_dist(feat_list, labels_list)

#         if self.rank != 0:
#             return None

#         dims = [d for d in self._get_eval_dims() if d <= self.pca_components.shape[0]]

#         if len(dims) == 0:
#             raise ValueError("No valid PCA dimensions. PCA components={} dims={}".format(
#                 self.pca_components.shape[0], self._get_eval_dims()
#             ))

#         labels_all = labels_all.cpu()
#         logging.info("========== OpenShape PCA: {} ==========".format(dataset_name))
#         logging.info("Feature dim: shape={} text={} | eval dims={}".format(
#             tuple(feat_all.shape), tuple(clip_text_feat.shape), dims
#         ))

#         results = {}
#         for d in dims:
#             shape_d = self._apply_pca(feat_all, d)
#             text_d = self._apply_pca(clip_text_feat, d)

#             shape_d = F.normalize(shape_d, dim=1)
#             text_d = F.normalize(text_d, dim=1)
#             logits_all = shape_d @ text_d.T

#             stats = self._compute_zero_shot_stats(logits_all, labels_all, num_classes)
#             results[d] = {k: float(v.item()) for k, v in stats.items()}

#             var_preserved = self._get_pca_explained_variance(d)
#             results[d]["explained_variance_preserved"] = var_preserved

#             if var_preserved is not None:
#                 logging.info(
#                     "OpenShape PCA | {} | D={}: explained_variance_preserved: {:.2f}% "
#                     "overall_acc: {:.4f} class_acc: {:.4f} "
#                     "top1_acc: {:.4f} top3_acc: {:.4f} top5_acc: {:.4f}".format(
#                         dataset_name,
#                         d,
#                         var_preserved,
#                         results[d]["overall_acc"],
#                         results[d]["class_acc"],
#                         results[d]["top1"],
#                         results[d]["top3"],
#                         results[d]["top5"],
#                     )
#                 )
#             else:
#                 logging.info(
#                     "OpenShape PCA | {} | D={}: overall_acc: {:.4f} class_acc: {:.4f} "
#                     "top1_acc: {:.4f} top3_acc: {:.4f} top5_acc: {:.4f}".format(
#                         dataset_name,
#                         d,
#                         results[d]["overall_acc"],
#                         results[d]["class_acc"],
#                         results[d]["top1"],
#                         results[d]["top3"],
#                         results[d]["top5"],
#                     )
#                 )

#         return results

#     def test_modelnet40_pca(self):
#         return self.test_zero_shot_pca(self.modelnet40_loader, "ModelNet40", 40)

#     def test_objaverse_lvis_pca(self):
#         return self.test_zero_shot_pca(self.objaverse_lvis_loader, "ObjaverseLVIS", 1156)

#     def test_scanobjectnn_pca(self):
#         return self.test_zero_shot_pca(self.scanobjectnn_loader, "ScanObjectNN", 15)

#     def train(self):
#         for epoch in range(self.epoch, self.config.training.max_epoch):
#             self.epoch = epoch
#             if self.rank == 0:
#                 logging.info("Epoch: {}".format(self.epoch))
#             self.train_one_epoch()
#             if epoch > self.config.training.test_epoch:
#                 self.test_objaverse_lvis()
#                 # self.test_modelnet40()
#                 self.test_objaverse_lvis()
#             if self.rank == 0:
#                 self.save_model('latest')
#             if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
#                 self.save_model('epoch_{}'.format(self.epoch))


#     def test_modelnet40(self):
#         self.model.eval()
#         if self.config.training.use_text_proj:
#             self.text_proj.eval()
#         clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).cuda()
#         if self.config.training.use_text_proj:
#             clip_text_feat = self.text_proj(clip_text_feat)
#         per_cat_correct = torch.zeros(40).cuda()
#         per_cat_count = torch.zeros(40).cuda()
#         category2idx = self.modelnet40_loader.dataset.category2idx
#         idx2category = {v: k for k, v in category2idx.items()}

#         logits_all = []
#         labels_all = []
#         with torch.no_grad():
#             for data in self.modelnet40_loader:
#                 if not self.config.model.get("use_dense", False):
#                     pred_feat = self.model(data['xyz'], data['features'], \
#                                            device=self.config.device, \
#                                            quantization_size=self.config.model.voxel_size)
#                 else:
#                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])

#                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
#                 labels = data['category'].to(self.config.device)
#                 logits_all.append(logits.detach())
#                 labels_all.append(labels)

#         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "modelnet40_dir"),
#                                                     logits_all, labels_all)

#         if self.rank == 0:

#             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

#             for i in range(40):
#                 idx = (labels_all == i)
#                 if idx.sum() > 0:
#                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
#                     per_cat_count[i] = idx.sum()

#             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
#             per_cat_acc = per_cat_correct / per_cat_count
#             # for i in range(40):
#             #    print(idx2category[i], per_cat_acc[i])

#             if overall_acc > self.best_modelnet40_overall_acc:
#                 self.best_modelnet40_overall_acc = overall_acc
#                 # self.save_model('best_modelnet40_overall')
#             if per_cat_acc.mean() > self.best_modelnet40_class_acc:
#                 self.best_modelnet40_class_acc = per_cat_acc.mean()
#                 # self.save_model('best_modelnet40_class')

#             logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
#                                                                                              self.best_modelnet40_overall_acc,
#                                                                                              per_cat_acc.mean(),
#                                                                                              self.best_modelnet40_class_acc))
#             logging.info(
#                 'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
#                                                                                     topk_acc[1].item(),
#                                                                                     topk_acc[2].item()))
#             # wandb.log({"test/epoch": self.epoch,
#             #            "test/step": self.step,
#             #            "test/ModelNet40_overall_acc": overall_acc,
#             #            "test/ModelNet40_class_acc": per_cat_acc.mean(),
#             #            "test/top3_acc": topk_acc[1],
#             #            "test/top5_acc": topk_acc[2], })

#     def test_objaverse_lvis(self):
#         self.model.eval()
#         if self.config.training.use_text_proj:
#             self.text_proj.eval()
#         clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).cuda()
#         if self.config.training.use_text_proj:
#             clip_text_feat = self.text_proj(clip_text_feat)
#         per_cat_correct = torch.zeros(1156).cuda()
#         per_cat_count = torch.zeros(1156).cuda()
#         category2idx = self.objaverse_lvis_loader.dataset.category2idx
#         idx2category = {v: k for k, v in category2idx.items()}

#         logits_all = []
#         labels_all = []
#         with torch.no_grad():
#             for data in self.objaverse_lvis_loader:
#                 if not self.config.model.get("use_dense", False):
#                     pred_feat = self.model(data['xyz'], data['features'], \
#                                            device=self.config.device, \
#                                            quantization_size=self.config.model.voxel_size)
#                 else:
#                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
#                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
#                 labels = data['category'].to(self.config.device)
#                 logits_all.append(logits.detach())
#                 labels_all.append(labels)

#         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "objaverse_dir"),
#                                                     logits_all, labels_all)

#         if self.rank == 0:
#             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

#             # calculate per class accuracy
#             for i in torch.unique(labels_all):
#                 idx = (labels_all == i)
#                 if idx.sum() > 0:
#                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
#                     per_cat_count[i] = idx.sum()

#             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
#             per_cat_acc = per_cat_correct / per_cat_count

#             if overall_acc > self.best_lvis_acc:
#                 self.best_lvis_acc = overall_acc
#                 # self.save_model('best_lvis')

#             logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
#             logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
#                                                                                                 topk_acc[1].item(),
#                                                                                                 topk_acc[2].item()))
#             # wandb.log({"test_lvis/epoch": self.epoch,
#             #            "test_lvis/step": self.step,
#             #            "test_lvis/overall_acc": overall_acc,
#             #            "test_lvis/class_acc": per_cat_acc.mean(),
#             #            "test_lvis/top3_acc": topk_acc[1],
#             #            "test_lvis/top5_acc": topk_acc[2], })

#     def test_scanobjectnn(self):
#         self.model.eval()
#         if self.config.training.use_text_proj:
#             self.text_proj.eval()
#         clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
#         if self.config.training.use_text_proj:
#             clip_text_feat = self.text_proj(clip_text_feat)
#         per_cat_correct = torch.zeros(15).to(self.config.device)
#         per_cat_count = torch.zeros(15).to(self.config.device)
#         category2idx = self.scanobjectnn_loader.dataset.category2idx
#         idx2category = {v: k for k, v in category2idx.items()}

#         logits_all = []
#         labels_all = []
#         with torch.no_grad():
#             for data in self.scanobjectnn_loader:
#                 if not self.config.model.get("use_dense", False):
#                     pred_feat = self.model(data['xyz'], data['features'], \
#                                            device=self.config.device, \
#                                            quantization_size=self.config.model.voxel_size)
#                 else:
#                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
#                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
#                 labels = data['category'].to(self.config.device)
#                 logits_all.append(logits.detach())
#                 labels_all.append(labels)

#         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scanobjectnn_dir"),
#                                                     logits_all, labels_all)

#         if self.rank == 0:

#             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

#             # calculate per class accuracy
#             for i in range(15):
#                 idx = (labels_all == i)
#                 if idx.sum() > 0:
#                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
#                     per_cat_count[i] = idx.sum()

#             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
#             per_cat_acc = per_cat_correct / per_cat_count

#             logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
#             logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
#                                                                                                topk_acc[1].item(),
#                                                                                                topk_acc[2].item()))
#             # wandb.log({"test_scanobjectnn/epoch": self.epoch,
#             #            "test_scanobjectnn/step": self.step,
#             #            "test_scanobjectnn/overall_acc": overall_acc,
#             #            "test_scanobjectnn/class_acc": per_cat_acc.mean(),
#             #            "test_scanobjectnn/top3_acc": topk_acc[1],
#             #            "test_scanobjectnn/top5_acc": topk_acc[2], })


#     def test_scannet(self, scannet_loader):
#         self.model.eval()
#         if self.config.training.use_text_proj:
#             self.text_proj.eval()
#         clip_text_feat = torch.from_numpy(scannet_loader.dataset.clip_cat_feat).to(self.config.device)
#         if self.config.training.use_text_proj:
#             clip_text_feat = self.text_proj(clip_text_feat)
#         per_cat_correct = torch.zeros(19).to(self.config.device)
#         per_cat_count = torch.zeros(19).to(self.config.device)
#         category2idx = scannet_loader.dataset.category2idx
#         idx2category = {v: k for k, v in category2idx.items()}

#         logits_all = []
#         labels_all = []
#         with torch.no_grad():
#             for data in tqdm(scannet_loader):
#                 if not self.config.model.get("use_dense", False):
#                     pred_feat = self.model(data['xyz'], data['features'], \
#                                            device=self.config.device, \
#                                            quantization_size=self.config.model.voxel_size)
#                 else:
#                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
#                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
#                 labels = data['category'].to(self.config.device)
#                 logits_all.append(logits.detach())
#                 labels_all.append(labels)

#         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scannet_dir"),
#                                                     logits_all, labels_all)
#         if self.rank == 0:

#             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

#             # calculate per class accuracy
#             for i in range(19):
#                 idx = (logits_all == i)
#                 if idx.sum() > 0:
#                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
#                     per_cat_count[i] = idx.sum()

#             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
#             per_cat_acc = per_cat_correct / per_cat_count


#             logging.info('Test Scannet: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
#             logging.info('Test Scannet: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
#                                                                                            topk_acc[1].item(),
#                                                                                                topk_acc[2].item()))

# # Backward-compatible alias used by your main/test script.
# # TrainerOpenShape = Trainer

# # import logging
# # import os

# # import numpy as np
# # import torch
# # import torch.distributed.nn
# # import torch.nn.functional as F
# # import wandb
# # import torch.distributed as dist
# # from trainers.trainer_utils import merge_results_dist
# # from tqdm import tqdm
# # from trainers.trainer_utils import merge_two_branch_results_dist


# # class TrainerOpenShape(object):
# #     def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, train_loader, \
# #                  modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None, clip_adapter=None):
# #         self.rank = rank
# #         self.config = config
# #         self.model = model
# #         self.logit_scale = logit_scale
# #         self.image_proj = image_proj
# #         self.text_proj = text_proj
# #         self.optimizer = optimizer
# #         self.scheduler = scheduler
# #         self.train_loader = train_loader
# #         self.modelnet40_loader = modelnet40_loader
# #         self.objaverse_lvis_loader = objaverse_lvis_loader
# #         self.scanobjectnn_loader = scanobjectnn_loader
# #         self.epoch = 0
# #         self.step = 0
# #         self.clip_adapter = clip_adapter
# #         self.best_img_contras_acc = 0
# #         self.best_text_contras_acc = 0
# #         self.best_modelnet40_overall_acc = 0
# #         self.best_modelnet40_class_acc = 0
# #         self.best_lvis_acc = 0
# #         self.config.ngpu = dist.get_world_size()

# #     def _cfg_get(self, obj, key, default=None):
# #         """Read config values from dict/OmegaConf-like objects safely."""
# #         if obj is None:
# #             return default
# #         if isinstance(obj, dict):
# #             return obj.get(key, default)
# #         try:
# #             return obj.get(key, default)
# #         except Exception:
# #             return getattr(obj, key, default)

# #     def _get_module(self, module):
# #         """Return the wrapped module when using DataParallel/DDP."""
# #         return module.module if hasattr(module, "module") else module

# #     def _clean_ddp_state_dict(self, state_dict):
# #         """Remove the 'module.' prefix from checkpoints saved with DDP."""
# #         if state_dict is None:
# #             return None
# #         return {
# #             k[len("module."):] if k.startswith("module.") else k: v
# #             for k, v in state_dict.items()
# #         }

# #     def _load_state_dict_clean(self, module, state_dict, strict=True, name="module"):
# #         """Load a state_dict robustly, handling DDP/non-DDP checkpoints."""
# #         state_dict = self._clean_ddp_state_dict(state_dict)
# #         msg = self._get_module(module).load_state_dict(state_dict, strict=strict)
# #         logging.info("Loaded {} state_dict: {}".format(name, msg))
# #         return msg

# #     def load_from_checkpoint(self, path, resume=True, strict=True):
# #         """
# #         Load OpenShape checkpoint.

# #         Use resume=False for evaluation-only compression baselines, e.g.
# #         OpenShape truncated and OpenShape PCA. This avoids requiring optimizer
# #         and scheduler states when you only want to test a frozen checkpoint.
# #         """
# #         checkpoint = torch.load(path, map_location="cpu")

# #         self._load_state_dict_clean(
# #             self.model,
# #             checkpoint["state_dict"],
# #             strict=strict,
# #             name="model",
# #         )

# #         if "logit_scale" in checkpoint and checkpoint["logit_scale"] is not None:
# #             self._load_state_dict_clean(
# #                 self.logit_scale,
# #                 checkpoint["logit_scale"],
# #                 strict=strict,
# #                 name="logit_scale",
# #             )

# #         if (
# #             self.config.training.use_text_proj
# #             and "text_proj" in checkpoint
# #             and checkpoint["text_proj"] is not None
# #         ):
# #             self._load_state_dict_clean(
# #                 self.text_proj,
# #                 checkpoint["text_proj"],
# #                 strict=strict,
# #                 name="text_proj",
# #             )

# #         if (
# #             self.config.training.use_image_proj
# #             and "image_proj" in checkpoint
# #             and checkpoint["image_proj"] is not None
# #         ):
# #             self._load_state_dict_clean(
# #                 self.image_proj,
# #                 checkpoint["image_proj"],
# #                 strict=strict,
# #                 name="image_proj",
# #             )

# #         if resume:
# #             if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
# #                 self.optimizer.load_state_dict(checkpoint["optimizer"])
# #             if self.config.training.use_openclip_optimizer_scheduler == False:
# #                 if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
# #                     self.scheduler.load_state_dict(checkpoint["scheduler"])
# #             self.epoch = checkpoint.get("epoch", 0)
# #             self.step = checkpoint.get("step", 0)
# #         else:
# #             self.epoch = 0
# #             self.step = 0

# #         logging.info("Loaded checkpoint from {}".format(path))
# #         logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

# #     def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
# #         if self.config.ngpu > 1:
# #             feat1 = F.normalize(feat1, dim=1)
# #             feat2 = F.normalize(feat2, dim=1)
# #             all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
# #             all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
# #             logits = logit_scale * all_feat1 @ all_feat2.T
# #         else:
# #             logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
# #         if mask is not None:
# #             logits = logits * mask
# #         labels = torch.arange(logits.shape[0]).to(self.config.device)
# #         accuracy = (logits.argmax(dim=1) == labels).float().mean()
# #         loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
# #         return loss, accuracy

# #     def train_one_epoch(self):
# #         self.model.train()
# #         if self.config.training.use_text_proj: #False
# #             self.text_proj.train()
# #         if self.config.training.use_image_proj: #False
# #             self.image_proj.train()

# #         text_contras_acc_list = []
# #         img_contras_acc_list = []
# #         if self.config.training.use_mask:
# #             k = self.config.dataset.negative_sample_num
# #             s = self.config.dataset.train_batch_size
# #             mask1 = np.eye(k * s).astype(np.bool)
# #             mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
# #             mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

# #         for data in tqdm(self.train_loader):
# #             self.step += 1
# #             self.optimizer.zero_grad()
# #             loss = 0
# #             if not self.config.model.get("use_dense", False):
# #                 pred_feat = self.model(data['xyz'], data['features'], \
# #                                        device=self.config.device, \
# #                                        quantization_size=self.config.model.voxel_size)
# #             else:
# #                 pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# #             logit_scale = self.logit_scale(None)
# #             idx = data['has_text_idx']

# #             text_feat = torch.vstack(data['text_feat']).to(self.config.device)
# #             img_feat = torch.vstack(data['img_feat']).to(self.config.device)

# #             if self.config.training.use_mask:
# #                 img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
# #                 mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
# #                 mask = torch.logical_or(mask, mask_other).detach()
# #             else:
# #                 mask = None

# #             if self.clip_adapter is not None:
# #                 with torch.no_grad():
# #                     img_feat = self.clip_adapter(img_feat)


# #             if self.config.training.use_image_proj:
# #                 img_feat = self.image_proj(img_feat)
# #             img_contras_loss, img_contras_acc = self.contras_loss(pred_feat, img_feat, logit_scale=logit_scale,
# #                                                                   mask=mask)

# #             loss += img_contras_loss * self.config.training.lambda_img_contras


# #             if len(idx) > 0:
# #                 if self.config.training.use_text_proj:
# #                     text_feat = self.text_proj(text_feat)
# #                 text_contras_loss, text_contras_acc = self.contras_loss(pred_feat[idx], text_feat,
# #                                                                         logit_scale=logit_scale, mask=mask)

# #                 loss += text_contras_loss * self.config.training.lambda_text_contras



# #             text_contras_acc_list.append(text_contras_acc.item())
# #             img_contras_acc_list.append(img_contras_acc.item())
# #             loss.backward()
# #             self.optimizer.step()
# #             if self.config.training.use_openclip_optimizer_scheduler:
# #                 self.scheduler(self.step)
# #             else:
# #                 self.scheduler.step()

# #         if self.rank == 0:
# #             logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}' \
# #                          .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
# #                                  np.mean(img_contras_acc_list)))

# #     def save_model(self, name):
# #         torch.save({
# #             "state_dict": self.model.state_dict(),
# #             "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
# #             "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
# #             "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
# #             "optimizer": self.optimizer.state_dict(),
# #             "scheduler": self.scheduler.state_dict() if self.config.training.use_openclip_optimizer_scheduler == False else None,
# #             "epoch": self.epoch,
# #             "step": self.step,
# #             "best_img_contras_acc": self.best_img_contras_acc,
# #             "best_text_contras_acc": self.best_text_contras_acc,
# #             "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
# #             "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
# #             "best_lvis_acc": self.best_lvis_acc,
# #         }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

# #     def accuracy(self, output, target, topk=(1,)):
# #         """Computes the accuracy over the k top predictions for the specified values of k"""
# #         with torch.no_grad():
# #             maxk = max(topk)
# #             batch_size = target.size(0)

# #             _, pred = output.topk(maxk, 1, True, True)
# #             pred = pred.t()
# #             correct = pred.eq(target.reshape(1, -1).expand_as(pred))

# #             res = []
# #             for k in topk:
# #                 correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
# #                 res.append(correct_k.mul_(100.0 / batch_size))
# #             return res, correct

# #     # -------------------------------------------------------------------------
# #     # OpenShape compression baselines: truncated and PCA.
# #     # These methods are eval-only: the OpenShape encoder is frozen.
# #     # -------------------------------------------------------------------------

# #     def _get_eval_dims(self):
# #         """Dimensions used by compression baselines."""
# #         eval_cfg = self._cfg_get(self.config, "eval", None)

# #         dims = self._cfg_get(eval_cfg, "pca_dims", None)
# #         if dims is None:
# #             dims = self._cfg_get(eval_cfg, "truncated_dims", None)
# #         if dims is None:
# #             dims = self._cfg_get(self._cfg_get(self.config, "mrl", None), "nesting_dims", None)
# #         if dims is None:
# #             dims = self._cfg_get(self._cfg_get(self.config, "model", None), "nesting_list", None)
# #         if dims is None:
# #             dims = [10, 20, 40, 80, 160, 320, 640, 1280]

# #         return [int(d) for d in dims]

# #     def _forward_pointcloud(self, data):
# #         """Forward pass for the 3D encoder, shared by normal/truncated/PCA evaluation."""
# #         if not self.config.model.get("use_dense", False):
# #             return self.model(
# #                 data["xyz"],
# #                 data["features"],
# #                 device=self.config.device,
# #                 quantization_size=self.config.model.voxel_size,
# #             )
# #         return self.model(data["xyz_dense"], data["features_dense"])

# #     def _merge_features_dist(self, feat_list):
# #         """
# #         Merge variable-length feature tensors across DDP ranks.
# #         Returns the merged tensor only on rank 0; other ranks receive None.
# #         """
# #         feat_local = torch.cat(feat_list, dim=0).contiguous()

# #         if (not dist.is_available()) or (not dist.is_initialized()) or self.config.ngpu <= 1:
# #             return feat_local

# #         world_size = dist.get_world_size()
# #         device = feat_local.device

# #         local_n = torch.tensor([feat_local.shape[0]], device=device, dtype=torch.long)
# #         sizes_t = [torch.zeros_like(local_n) for _ in range(world_size)]
# #         dist.all_gather(sizes_t, local_n)

# #         sizes = [int(x.item()) for x in sizes_t]
# #         max_n = max(sizes)
# #         feat_dim = feat_local.shape[1]

# #         if feat_local.shape[0] < max_n:
# #             pad_n = max_n - feat_local.shape[0]
# #             feat_pad = torch.zeros(pad_n, feat_dim, device=device, dtype=feat_local.dtype)
# #             feat_local = torch.cat([feat_local, feat_pad], dim=0)

# #         gathered_feats = [torch.zeros_like(feat_local) for _ in range(world_size)]
# #         dist.all_gather(gathered_feats, feat_local)

# #         if self.rank == 0:
# #             return torch.cat(
# #                 [gathered_feats[r][:sizes[r]] for r in range(world_size)],
# #                 dim=0,
# #             )

# #         return None

# #     def _merge_features_labels_dist(self, feat_list, label_list):
# #         """
# #         Merge variable-length feature/label tensors across DDP ranks.
# #         Returns merged tensors only on rank 0; other ranks receive (None, None).
# #         """
# #         feat_local = torch.cat(feat_list, dim=0).contiguous()
# #         labels_local = torch.cat(label_list, dim=0).contiguous()

# #         if (not dist.is_available()) or (not dist.is_initialized()) or self.config.ngpu <= 1:
# #             return feat_local, labels_local

# #         world_size = dist.get_world_size()
# #         device = feat_local.device

# #         local_n = torch.tensor([feat_local.shape[0]], device=device, dtype=torch.long)
# #         sizes_t = [torch.zeros_like(local_n) for _ in range(world_size)]
# #         dist.all_gather(sizes_t, local_n)

# #         sizes = [int(x.item()) for x in sizes_t]
# #         max_n = max(sizes)
# #         feat_dim = feat_local.shape[1]

# #         if feat_local.shape[0] < max_n:
# #             pad_n = max_n - feat_local.shape[0]
# #             feat_pad = torch.zeros(pad_n, feat_dim, device=device, dtype=feat_local.dtype)
# #             label_pad = torch.zeros(pad_n, device=device, dtype=labels_local.dtype)
# #             feat_local = torch.cat([feat_local, feat_pad], dim=0)
# #             labels_local = torch.cat([labels_local, label_pad], dim=0)

# #         gathered_feats = [torch.zeros_like(feat_local) for _ in range(world_size)]
# #         gathered_labels = [torch.zeros_like(labels_local) for _ in range(world_size)]

# #         dist.all_gather(gathered_feats, feat_local)
# #         dist.all_gather(gathered_labels, labels_local)

# #         if self.rank == 0:
# #             feat_all = torch.cat(
# #                 [gathered_feats[r][:sizes[r]] for r in range(world_size)],
# #                 dim=0,
# #             )
# #             labels_all = torch.cat(
# #                 [gathered_labels[r][:sizes[r]] for r in range(world_size)],
# #                 dim=0,
# #             )
# #             return feat_all, labels_all

# #         return None, None

# #     def _compute_zero_shot_stats(self, logits_all, labels_all, num_classes):
# #         """Compute overall accuracy, mean class accuracy, and top-k accuracy."""
# #         device = logits_all.device
# #         labels_all = labels_all.to(device)

# #         per_cat_correct = torch.zeros(num_classes, device=device)
# #         per_cat_count = torch.zeros(num_classes, device=device)

# #         topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# #         for i in torch.unique(labels_all):
# #             idx = labels_all == i
# #             if idx.sum() > 0:
# #                 per_cat_correct[i] = (
# #                     logits_all[idx].argmax(dim=1) == labels_all[idx]
# #                 ).float().sum()
# #                 per_cat_count[i] = idx.sum()

# #         valid = per_cat_count > 0
# #         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# #         class_acc = (per_cat_correct[valid] / per_cat_count[valid]).mean()

# #         return {
# #             "overall_acc": overall_acc,
# #             "class_acc": class_acc,
# #             "top1": topk_acc[0],
# #             "top3": topk_acc[1],
# #             "top5": topk_acc[2],
# #         }

# #     def test_zero_shot_truncated(self, loader, dataset_name, num_classes):
# #         """
# #         OpenShape truncated baseline.

# #         Protocol:
# #         1) Extract full OpenShape 3D features, usually 1280-D.
# #         2) Merge features across DDP ranks once.
# #         3) For each d, evaluate normalize(feat[:, :d]) @ normalize(text[:, :d]).T.
# #         """
# #         self.model.eval()
# #         if self.config.training.use_text_proj:
# #             self.text_proj.eval()

# #         clip_text_feat = torch.from_numpy(loader.dataset.clip_cat_feat).to(self.config.device)
# #         if self.config.training.use_text_proj:
# #             clip_text_feat = self.text_proj(clip_text_feat)

# #         feat_list = []
# #         labels_list = []

# #         with torch.no_grad():
# #             for data in tqdm(loader, desc="Extracting {} features".format(dataset_name)):
# #                 pred_feat = self._forward_pointcloud(data)
# #                 labels = data["category"].to(self.config.device)
# #                 feat_list.append(pred_feat.detach())
# #                 labels_list.append(labels.detach())

# #         feat_all, labels_all = self._merge_features_labels_dist(feat_list, labels_list)

# #         if self.rank != 0:
# #             return None

# #         clip_text_feat = clip_text_feat.to(feat_all.device)
# #         max_dim = min(feat_all.shape[1], clip_text_feat.shape[1])
# #         dims = [d for d in self._get_eval_dims() if d <= max_dim]

# #         if len(dims) == 0:
# #             raise ValueError("No valid eval dimensions. max_dim={} dims={}".format(max_dim, self._get_eval_dims()))

# #         logging.info("========== OpenShape truncated: {} ==========".format(dataset_name))
# #         logging.info("Feature dim: shape={} text={} | eval dims={}".format(
# #             tuple(feat_all.shape), tuple(clip_text_feat.shape), dims
# #         ))

# #         results = {}
# #         for d in dims:
# #             shape_d = F.normalize(feat_all[:, :d], dim=1)
# #             text_d = F.normalize(clip_text_feat[:, :d], dim=1)
# #             logits_all = shape_d @ text_d.T

# #             stats = self._compute_zero_shot_stats(logits_all, labels_all, num_classes)
# #             results[d] = {k: float(v.item()) for k, v in stats.items()}

# #             logging.info(
# #                 "OpenShape truncated | {} | D={}: overall_acc: {:.4f} class_acc: {:.4f} "
# #                 "top1_acc: {:.4f} top3_acc: {:.4f} top5_acc: {:.4f}".format(
# #                     dataset_name,
# #                     d,
# #                     results[d]["overall_acc"],
# #                     results[d]["class_acc"],
# #                     results[d]["top1"],
# #                     results[d]["top3"],
# #                     results[d]["top5"],
# #                 )
# #             )

# #         return results

# #     def test_modelnet40_truncated(self):
# #         return self.test_zero_shot_truncated(self.modelnet40_loader, "ModelNet40", 40)

# #     def test_objaverse_lvis_truncated(self):
# #         return self.test_zero_shot_truncated(self.objaverse_lvis_loader, "ObjaverseLVIS", 1156)

# #     def test_scanobjectnn_truncated(self):
# #         return self.test_zero_shot_truncated(self.scanobjectnn_loader, "ScanObjectNN", 15)

# #     def fit_pca_from_train_loader(self):
# #         """
# #         Fit PCA on frozen 1280-D OpenShape training embeddings.

# #         PCA is fitted on rank 0 only. Evaluation also applies the projection on
# #         rank 0 after merging features, so other ranks do not need the PCA state.
# #         """
# #         self.model.eval()

# #         eval_cfg = self._cfg_get(self.config, "eval", None)
# #         max_samples = self._cfg_get(eval_cfg, "pca_fit_samples", 20000)
# #         seed = int(self._cfg_get(eval_cfg, "pca_seed", 0))
# #         pca_niter = int(self._cfg_get(eval_cfg, "pca_niter", 2))

# #         feat_list = []
# #         with torch.no_grad():
# #             for data in tqdm(self.train_loader, desc="Extracting train features for PCA"):
# #                 pred_feat = self._forward_pointcloud(data)
# #                 feat_list.append(pred_feat.detach())

# #         feat_all = self._merge_features_dist(feat_list)

# #         if self.rank != 0:
# #             return None

# #         feat_all = feat_all.float().cpu()

# #         if max_samples is not None and int(max_samples) > 0 and feat_all.shape[0] > int(max_samples):
# #             generator = torch.Generator(device="cpu")
# #             generator.manual_seed(seed)
# #             perm = torch.randperm(feat_all.shape[0], generator=generator)[: int(max_samples)]
# #             feat_all = feat_all[perm]

# #         dims = self._get_eval_dims()
# #         max_dim = min(max(dims), feat_all.shape[0], feat_all.shape[1])

# #         logging.info(
# #             "Fitting OpenShape PCA with n_components={} using {} samples.".format(
# #                 max_dim, feat_all.shape[0]
# #             )
# #         )

# #         mean = feat_all.mean(dim=0, keepdim=True)
# #         centered = feat_all - mean

# #         try:
# #             # V has shape [in_dim, max_dim]. Components are V.T.
# #             _, _, v = torch.pca_lowrank(centered, q=max_dim, center=False, niter=pca_niter)
# #             components = v[:, :max_dim].T.contiguous()
# #         except Exception as exc:
# #             logging.warning("torch.pca_lowrank failed ({}). Falling back to torch.linalg.svd.".format(exc))
# #             _, _, vh = torch.linalg.svd(centered, full_matrices=False)
# #             components = vh[:max_dim].contiguous()

# #         self.pca_mean = mean.squeeze(0).contiguous()
# #         self.pca_components = components.contiguous()

# #         cache_path = self._cfg_get(eval_cfg, "pca_cache_path", None)
# #         if cache_path is None:
# #             cache_path = os.path.join(self.config.ckpt_dir, "openshape_pca_fit.pt")

# #         torch.save(
# #             {
# #                 "mean": self.pca_mean,
# #                 "components": self.pca_components,
# #                 "dims": dims,
# #                 "fit_samples": feat_all.shape[0],
# #             },
# #             cache_path,
# #         )
# #         logging.info("Saved PCA cache to {}".format(cache_path))
# #         logging.info("Finished fitting PCA.")
# #         return cache_path

# #     def load_pca(self, path=None):
# #         """Load a previously fitted PCA cache."""
# #         eval_cfg = self._cfg_get(self.config, "eval", None)
# #         if path is None:
# #             path = self._cfg_get(eval_cfg, "pca_cache_path", None)
# #         if path is None:
# #             path = os.path.join(self.config.ckpt_dir, "openshape_pca_fit.pt")

# #         pca = torch.load(path, map_location="cpu")
# #         self.pca_mean = pca["mean"].float().contiguous()
# #         self.pca_components = pca["components"].float().contiguous()
# #         logging.info("Loaded PCA cache from {}".format(path))

# #     def _apply_pca(self, feat, dim):
# #         """Apply fitted PCA to a feature tensor and return an [N, dim] CPU tensor."""
# #         if not hasattr(self, "pca_mean") or not hasattr(self, "pca_components"):
# #             raise RuntimeError("PCA is not fitted. Call fit_pca_from_train_loader() or load_pca() first.")

# #         if dim > self.pca_components.shape[0]:
# #             raise ValueError("Requested dim={} but PCA has only {} components.".format(
# #                 dim, self.pca_components.shape[0]
# #             ))

# #         feat = feat.float().cpu()
# #         mean = self.pca_mean.float().cpu()
# #         components = self.pca_components[:dim].float().cpu()
# #         return (feat - mean) @ components.T

# #     def test_zero_shot_pca(self, loader, dataset_name, num_classes):
# #         """
# #         OpenShape + PCA baseline.

# #         The same PCA projection is applied to shape and text embeddings, then
# #         features are L2-normalized and evaluated with cosine similarity.
# #         """
# #         self.model.eval()
# #         if self.config.training.use_text_proj:
# #             self.text_proj.eval()

# #         clip_text_feat = torch.from_numpy(loader.dataset.clip_cat_feat).to(self.config.device)
# #         if self.config.training.use_text_proj:
# #             clip_text_feat = self.text_proj(clip_text_feat)

# #         feat_list = []
# #         labels_list = []

# #         with torch.no_grad():
# #             for data in tqdm(loader, desc="Extracting {} features for PCA eval".format(dataset_name)):
# #                 pred_feat = self._forward_pointcloud(data)
# #                 labels = data["category"].to(self.config.device)
# #                 feat_list.append(pred_feat.detach())
# #                 labels_list.append(labels.detach())

# #         feat_all, labels_all = self._merge_features_labels_dist(feat_list, labels_list)

# #         if self.rank != 0:
# #             return None

# #         dims = [d for d in self._get_eval_dims() if d <= self.pca_components.shape[0]]

# #         if len(dims) == 0:
# #             raise ValueError("No valid PCA dimensions. PCA components={} dims={}".format(
# #                 self.pca_components.shape[0], self._get_eval_dims()
# #             ))

# #         labels_all = labels_all.cpu()
# #         logging.info("========== OpenShape PCA: {} ==========".format(dataset_name))
# #         logging.info("Feature dim: shape={} text={} | eval dims={}".format(
# #             tuple(feat_all.shape), tuple(clip_text_feat.shape), dims
# #         ))

# #         results = {}
# #         for d in dims:
# #             shape_d = self._apply_pca(feat_all, d)
# #             text_d = self._apply_pca(clip_text_feat, d)

# #             shape_d = F.normalize(shape_d, dim=1)
# #             text_d = F.normalize(text_d, dim=1)
# #             logits_all = shape_d @ text_d.T

# #             stats = self._compute_zero_shot_stats(logits_all, labels_all, num_classes)
# #             results[d] = {k: float(v.item()) for k, v in stats.items()}

# #             logging.info(
# #                 "OpenShape PCA | {} | D={}: overall_acc: {:.4f} class_acc: {:.4f} "
# #                 "top1_acc: {:.4f} top3_acc: {:.4f} top5_acc: {:.4f}".format(
# #                     dataset_name,
# #                     d,
# #                     results[d]["overall_acc"],
# #                     results[d]["class_acc"],
# #                     results[d]["top1"],
# #                     results[d]["top3"],
# #                     results[d]["top5"],
# #                 )
# #             )

# #         return results

# #     def test_modelnet40_pca(self):
# #         return self.test_zero_shot_pca(self.modelnet40_loader, "ModelNet40", 40)

# #     def test_objaverse_lvis_pca(self):
# #         return self.test_zero_shot_pca(self.objaverse_lvis_loader, "ObjaverseLVIS", 1156)

# #     def test_scanobjectnn_pca(self):
# #         return self.test_zero_shot_pca(self.scanobjectnn_loader, "ScanObjectNN", 15)

# #     def train(self):
# #         for epoch in range(self.epoch, self.config.training.max_epoch):
# #             self.epoch = epoch
# #             if self.rank == 0:
# #                 logging.info("Epoch: {}".format(self.epoch))
# #             self.train_one_epoch()
# #             if epoch > self.config.training.test_epoch:
# #                 self.test_objaverse_lvis()
# #                 # self.test_modelnet40()
# #                 self.test_objaverse_lvis()
# #             if self.rank == 0:
# #                 self.save_model('latest')
# #             if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
# #                 self.save_model('epoch_{}'.format(self.epoch))


# #     def test_modelnet40(self):
# #         self.model.eval()
# #         if self.config.training.use_text_proj:
# #             self.text_proj.eval()
# #         clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).cuda()
# #         if self.config.training.use_text_proj:
# #             clip_text_feat = self.text_proj(clip_text_feat)
# #         per_cat_correct = torch.zeros(40).cuda()
# #         per_cat_count = torch.zeros(40).cuda()
# #         category2idx = self.modelnet40_loader.dataset.category2idx
# #         idx2category = {v: k for k, v in category2idx.items()}

# #         logits_all = []
# #         labels_all = []
# #         with torch.no_grad():
# #             for data in self.modelnet40_loader:
# #                 if not self.config.model.get("use_dense", False):
# #                     pred_feat = self.model(data['xyz'], data['features'], \
# #                                            device=self.config.device, \
# #                                            quantization_size=self.config.model.voxel_size)
# #                 else:
# #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])

# #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# #                 labels = data['category'].to(self.config.device)
# #                 logits_all.append(logits.detach())
# #                 labels_all.append(labels)

# #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "modelnet40_dir"),
# #                                                     logits_all, labels_all)

# #         if self.rank == 0:

# #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# #             for i in range(40):
# #                 idx = (labels_all == i)
# #                 if idx.sum() > 0:
# #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# #                     per_cat_count[i] = idx.sum()

# #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# #             per_cat_acc = per_cat_correct / per_cat_count
# #             # for i in range(40):
# #             #    print(idx2category[i], per_cat_acc[i])

# #             if overall_acc > self.best_modelnet40_overall_acc:
# #                 self.best_modelnet40_overall_acc = overall_acc
# #                 # self.save_model('best_modelnet40_overall')
# #             if per_cat_acc.mean() > self.best_modelnet40_class_acc:
# #                 self.best_modelnet40_class_acc = per_cat_acc.mean()
# #                 # self.save_model('best_modelnet40_class')

# #             logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
# #                                                                                              self.best_modelnet40_overall_acc,
# #                                                                                              per_cat_acc.mean(),
# #                                                                                              self.best_modelnet40_class_acc))
# #             logging.info(
# #                 'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# #                                                                                     topk_acc[1].item(),
# #                                                                                     topk_acc[2].item()))
# #             # wandb.log({"test/epoch": self.epoch,
# #             #            "test/step": self.step,
# #             #            "test/ModelNet40_overall_acc": overall_acc,
# #             #            "test/ModelNet40_class_acc": per_cat_acc.mean(),
# #             #            "test/top3_acc": topk_acc[1],
# #             #            "test/top5_acc": topk_acc[2], })

# #     def test_objaverse_lvis(self):
# #         self.model.eval()
# #         if self.config.training.use_text_proj:
# #             self.text_proj.eval()
# #         clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).cuda()
# #         if self.config.training.use_text_proj:
# #             clip_text_feat = self.text_proj(clip_text_feat)
# #         per_cat_correct = torch.zeros(1156).cuda()
# #         per_cat_count = torch.zeros(1156).cuda()
# #         category2idx = self.objaverse_lvis_loader.dataset.category2idx
# #         idx2category = {v: k for k, v in category2idx.items()}

# #         logits_all = []
# #         labels_all = []
# #         with torch.no_grad():
# #             for data in self.objaverse_lvis_loader:
# #                 if not self.config.model.get("use_dense", False):
# #                     pred_feat = self.model(data['xyz'], data['features'], \
# #                                            device=self.config.device, \
# #                                            quantization_size=self.config.model.voxel_size)
# #                 else:
# #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# #                 labels = data['category'].to(self.config.device)
# #                 logits_all.append(logits.detach())
# #                 labels_all.append(labels)

# #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "objaverse_dir"),
# #                                                     logits_all, labels_all)

# #         if self.rank == 0:
# #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# #             # calculate per class accuracy
# #             for i in torch.unique(labels_all):
# #                 idx = (labels_all == i)
# #                 if idx.sum() > 0:
# #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# #                     per_cat_count[i] = idx.sum()

# #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# #             per_cat_acc = per_cat_correct / per_cat_count

# #             if overall_acc > self.best_lvis_acc:
# #                 self.best_lvis_acc = overall_acc
# #                 # self.save_model('best_lvis')

# #             logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# #             logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# #                                                                                                 topk_acc[1].item(),
# #                                                                                                 topk_acc[2].item()))
# #             # wandb.log({"test_lvis/epoch": self.epoch,
# #             #            "test_lvis/step": self.step,
# #             #            "test_lvis/overall_acc": overall_acc,
# #             #            "test_lvis/class_acc": per_cat_acc.mean(),
# #             #            "test_lvis/top3_acc": topk_acc[1],
# #             #            "test_lvis/top5_acc": topk_acc[2], })

# #     def test_scanobjectnn(self):
# #         self.model.eval()
# #         if self.config.training.use_text_proj:
# #             self.text_proj.eval()
# #         clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
# #         if self.config.training.use_text_proj:
# #             clip_text_feat = self.text_proj(clip_text_feat)
# #         per_cat_correct = torch.zeros(15).to(self.config.device)
# #         per_cat_count = torch.zeros(15).to(self.config.device)
# #         category2idx = self.scanobjectnn_loader.dataset.category2idx
# #         idx2category = {v: k for k, v in category2idx.items()}

# #         logits_all = []
# #         labels_all = []
# #         with torch.no_grad():
# #             for data in self.scanobjectnn_loader:
# #                 if not self.config.model.get("use_dense", False):
# #                     pred_feat = self.model(data['xyz'], data['features'], \
# #                                            device=self.config.device, \
# #                                            quantization_size=self.config.model.voxel_size)
# #                 else:
# #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# #                 labels = data['category'].to(self.config.device)
# #                 logits_all.append(logits.detach())
# #                 labels_all.append(labels)

# #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scanobjectnn_dir"),
# #                                                     logits_all, labels_all)

# #         if self.rank == 0:

# #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# #             # calculate per class accuracy
# #             for i in range(15):
# #                 idx = (labels_all == i)
# #                 if idx.sum() > 0:
# #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# #                     per_cat_count[i] = idx.sum()

# #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# #             per_cat_acc = per_cat_correct / per_cat_count

# #             logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# #             logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# #                                                                                                topk_acc[1].item(),
# #                                                                                                topk_acc[2].item()))
# #             # wandb.log({"test_scanobjectnn/epoch": self.epoch,
# #             #            "test_scanobjectnn/step": self.step,
# #             #            "test_scanobjectnn/overall_acc": overall_acc,
# #             #            "test_scanobjectnn/class_acc": per_cat_acc.mean(),
# #             #            "test_scanobjectnn/top3_acc": topk_acc[1],
# #             #            "test_scanobjectnn/top5_acc": topk_acc[2], })


# #     def test_scannet(self, scannet_loader):
# #         self.model.eval()
# #         if self.config.training.use_text_proj:
# #             self.text_proj.eval()
# #         clip_text_feat = torch.from_numpy(scannet_loader.dataset.clip_cat_feat).to(self.config.device)
# #         if self.config.training.use_text_proj:
# #             clip_text_feat = self.text_proj(clip_text_feat)
# #         per_cat_correct = torch.zeros(19).to(self.config.device)
# #         per_cat_count = torch.zeros(19).to(self.config.device)
# #         category2idx = scannet_loader.dataset.category2idx
# #         idx2category = {v: k for k, v in category2idx.items()}

# #         logits_all = []
# #         labels_all = []
# #         with torch.no_grad():
# #             for data in tqdm(scannet_loader):
# #                 if not self.config.model.get("use_dense", False):
# #                     pred_feat = self.model(data['xyz'], data['features'], \
# #                                            device=self.config.device, \
# #                                            quantization_size=self.config.model.voxel_size)
# #                 else:
# #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# #                 labels = data['category'].to(self.config.device)
# #                 logits_all.append(logits.detach())
# #                 labels_all.append(labels)

# #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scannet_dir"),
# #                                                     logits_all, labels_all)
# #         if self.rank == 0:

# #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# #             # calculate per class accuracy
# #             for i in range(19):
# #                 idx = (logits_all == i)
# #                 if idx.sum() > 0:
# #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# #                     per_cat_count[i] = idx.sum()

# #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# #             per_cat_acc = per_cat_correct / per_cat_count


# #             logging.info('Test Scannet: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# #             logging.info('Test Scannet: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# #                                                                                            topk_acc[1].item(),
# #                                                                                                topk_acc[2].item()))

# # # Backward-compatible alias used by your main/test script.
# # # TrainerOpenShape = Trainer


# # # import logging
# # # import os

# # # import numpy as np
# # # import torch
# # # import torch.distributed.nn
# # # import torch.nn.functional as F
# # # import wandb
# # # import torch.distributed as dist
# # # from trainers.trainer_utils import merge_results_dist
# # # from tqdm import tqdm
# # # from trainers.trainer_utils import merge_two_branch_results_dist

# # # from sklearn.decomposition import PCA

# # # class TrainerOpenShape(object):
# # #     def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, train_loader, \
# # #                  modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None, clip_adapter=None):
# # #         self.rank = rank
# # #         self.config = config
# # #         self.model = model
# # #         self.logit_scale = logit_scale
# # #         self.image_proj = image_proj
# # #         self.text_proj = text_proj
# # #         self.optimizer = optimizer
# # #         self.scheduler = scheduler
# # #         self.train_loader = train_loader
# # #         self.modelnet40_loader = modelnet40_loader
# # #         self.objaverse_lvis_loader = objaverse_lvis_loader
# # #         self.scanobjectnn_loader = scanobjectnn_loader
# # #         self.epoch = 0
# # #         self.step = 0
# # #         self.clip_adapter = clip_adapter
# # #         self.best_img_contras_acc = 0
# # #         self.best_text_contras_acc = 0
# # #         self.best_modelnet40_overall_acc = 0
# # #         self.best_modelnet40_class_acc = 0
# # #         self.best_lvis_acc = 0
# # #         self.config.ngpu = dist.get_world_size()

# # #     def _get_module(self, module):
# # #         """Return the wrapped module when using DataParallel/DDP."""
# # #         return module.module if hasattr(module, "module") else module

# # #     def _clean_ddp_state_dict(self, state_dict):
# # #         """Remove the `module.` prefix from checkpoints saved with DDP."""
# # #         if state_dict is None:
# # #             return None
# # #         return {k[len("module."):] if k.startswith("module.") else k: v
# # #                 for k, v in state_dict.items()}

# # #     def _load_state_dict_clean(self, module, state_dict, strict=True, name="module"):
# # #         """Load a state dict robustly, handling DDP/non-DDP checkpoints."""
# # #         state_dict = self._clean_ddp_state_dict(state_dict)
# # #         msg = self._get_module(module).load_state_dict(state_dict, strict=strict)
# # #         logging.info("Loaded {} state_dict: {}".format(name, msg))
# # #         return msg

# # #     def load_from_checkpoint(self, path, resume=True, strict=True):
# # #         checkpoint = torch.load(path, map_location="cpu")

# # #         self._load_state_dict_clean(self.model, checkpoint['state_dict'], strict=strict, name="model")

# # #         if 'logit_scale' in checkpoint and checkpoint['logit_scale'] is not None:
# # #             self._load_state_dict_clean(self.logit_scale, checkpoint['logit_scale'], strict=strict, name="logit_scale")

# # #         if self.config.training.use_text_proj and 'text_proj' in checkpoint and checkpoint['text_proj'] is not None:
# # #             self._load_state_dict_clean(self.text_proj, checkpoint['text_proj'], strict=strict, name="text_proj")

# # #         if self.config.training.use_image_proj and 'image_proj' in checkpoint and checkpoint['image_proj'] is not None:
# # #             self._load_state_dict_clean(self.image_proj, checkpoint['image_proj'], strict=strict, name="image_proj")

# # #         # Use resume=False for evaluation-only baselines such as OpenShape truncated.
# # #         if resume:
# # #             if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
# # #                 self.optimizer.load_state_dict(checkpoint['optimizer'])
# # #             if self.config.training.use_openclip_optimizer_scheduler == False:
# # #                 if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
# # #                     self.scheduler.load_state_dict(checkpoint['scheduler'])
# # #             self.epoch = checkpoint.get('epoch', 0)
# # #             self.step = checkpoint.get('step', 0)
# # #         else:
# # #             self.epoch = 0
# # #             self.step = 0

# # #         logging.info("Loaded checkpoint from {}".format(path))
# # #         logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

# # #     def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
# # #         if self.config.ngpu > 1:
# # #             feat1 = F.normalize(feat1, dim=1)
# # #             feat2 = F.normalize(feat2, dim=1)
# # #             all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
# # #             all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
# # #             logits = logit_scale * all_feat1 @ all_feat2.T
# # #         else:
# # #             logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
# # #         if mask is not None:
# # #             logits = logits * mask
# # #         labels = torch.arange(logits.shape[0]).to(self.config.device)
# # #         accuracy = (logits.argmax(dim=1) == labels).float().mean()
# # #         loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
# # #         return loss, accuracy

# # #     def train_one_epoch(self):
# # #         self.model.train()
# # #         if self.config.training.use_text_proj: #False
# # #             self.text_proj.train()
# # #         if self.config.training.use_image_proj: #False
# # #             self.image_proj.train()

# # #         text_contras_acc_list = []
# # #         img_contras_acc_list = []
# # #         if self.config.training.use_mask:
# # #             k = self.config.dataset.negative_sample_num
# # #             s = self.config.dataset.train_batch_size
# # #             mask1 = np.eye(k * s).astype(np.bool)
# # #             mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
# # #             mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

# # #         for data in tqdm(self.train_loader):
# # #             self.step += 1
# # #             self.optimizer.zero_grad()
# # #             loss = 0
# # #             if not self.config.model.get("use_dense", False):
# # #                 pred_feat = self.model(data['xyz'], data['features'], \
# # #                                        device=self.config.device, \
# # #                                        quantization_size=self.config.model.voxel_size)
# # #             else:
# # #                 pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # #             logit_scale = self.logit_scale(None)
# # #             idx = data['has_text_idx']

# # #             text_feat = torch.vstack(data['text_feat']).to(self.config.device)
# # #             img_feat = torch.vstack(data['img_feat']).to(self.config.device)

# # #             if self.config.training.use_mask:
# # #                 img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
# # #                 mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
# # #                 mask = torch.logical_or(mask, mask_other).detach()
# # #             else:
# # #                 mask = None

# # #             if self.clip_adapter is not None:
# # #                 with torch.no_grad():
# # #                     img_feat = self.clip_adapter(img_feat)


# # #             if self.config.training.use_image_proj:
# # #                 img_feat = self.image_proj(img_feat)
# # #             img_contras_loss, img_contras_acc = self.contras_loss(pred_feat, img_feat, logit_scale=logit_scale,
# # #                                                                   mask=mask)

# # #             loss += img_contras_loss * self.config.training.lambda_img_contras


# # #             if len(idx) > 0:
# # #                 if self.config.training.use_text_proj:
# # #                     text_feat = self.text_proj(text_feat)
# # #                 text_contras_loss, text_contras_acc = self.contras_loss(pred_feat[idx], text_feat,
# # #                                                                         logit_scale=logit_scale, mask=mask)

# # #                 loss += text_contras_loss * self.config.training.lambda_text_contras



# # #             text_contras_acc_list.append(text_contras_acc.item())
# # #             img_contras_acc_list.append(img_contras_acc.item())
# # #             loss.backward()
# # #             self.optimizer.step()
# # #             if self.config.training.use_openclip_optimizer_scheduler:
# # #                 self.scheduler(self.step)
# # #             else:
# # #                 self.scheduler.step()

# # #         if self.rank == 0:
# # #             logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}' \
# # #                          .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
# # #                                  np.mean(img_contras_acc_list)))

# # #     def save_model(self, name):
# # #         torch.save({
# # #             "state_dict": self.model.state_dict(),
# # #             "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
# # #             "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
# # #             "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
# # #             "optimizer": self.optimizer.state_dict(),
# # #             "scheduler": self.scheduler.state_dict() if self.config.training.use_openclip_optimizer_scheduler == False else None,
# # #             "epoch": self.epoch,
# # #             "step": self.step,
# # #             "best_img_contras_acc": self.best_img_contras_acc,
# # #             "best_text_contras_acc": self.best_text_contras_acc,
# # #             "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
# # #             "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
# # #             "best_lvis_acc": self.best_lvis_acc,
# # #         }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

# # #     def accuracy(self, output, target, topk=(1,)):
# # #         """Computes the accuracy over the k top predictions for the specified values of k"""
# # #         with torch.no_grad():
# # #             maxk = max(topk)
# # #             batch_size = target.size(0)

# # #             _, pred = output.topk(maxk, 1, True, True)
# # #             pred = pred.t()
# # #             correct = pred.eq(target.reshape(1, -1).expand_as(pred))

# # #             res = []
# # #             for k in topk:
# # #                 correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
# # #                 res.append(correct_k.mul_(100.0 / batch_size))
# # #             return res, correct

# # #     def _cfg_get(self, obj, key, default=None):
# # #         """Works with dict-like configs and EasyDict/OmegaConf-style configs."""
# # #         if obj is None:
# # #             return default
# # #         if hasattr(obj, "get"):
# # #             return obj.get(key, default)
# # #         return getattr(obj, key, default)
    
    
# # #     def _get_truncated_dims(self):
# # #         """Dimensions used for the OpenShape truncated baseline."""

# # #         default_dims = [10, 20, 40, 80, 160, 320, 640, 1280]

# # #         # fallback: config.mrl.nesting_dims
# # #         if hasattr(self.config, "mrl") and self._cfg_get(self.config.mrl, "nesting_dims", None) is not None:
# # #             default_dims = self.config.mrl.nesting_dims

# # #         dims = None

# # #         # Priority 1: config.eval.truncated_dims
# # #         eval_cfg = self._cfg_get(self.config, "eval", None)
# # #         if eval_cfg is not None:
# # #             dims = self._cfg_get(eval_cfg, "truncated_dims", None)

# # #         # Priority 2: config.model.truncated_dims
# # #         if dims is None:
# # #             dims = self._cfg_get(self.config.model, "truncated_dims", None)

# # #         # Priority 3: config.model.nesting_list
# # #         if dims is None:
# # #             dims = self._cfg_get(self.config.model, "nesting_list", None)

# # #         # Priority 4: config.mrl.nesting_dims
# # #         if dims is None:
# # #             dims = default_dims

# # #         return [int(d) for d in dims]

# # #     def _forward_pointcloud(self, data):
# # #         """Forward pass for the 3D encoder, shared by normal and truncated evaluation."""
# # #         if not self.config.model.get("use_dense", False):
# # #             return self.model(data['xyz'], data['features'],
# # #                               device=self.config.device,
# # #                               quantization_size=self.config.model.voxel_size)
# # #         return self.model(data['xyz_dense'], data['features_dense'])

# # #     def _compute_zero_shot_stats(self, logits_all, labels_all, num_classes):
# # #         """Compute overall acc, mean class acc, and top-k accuracy."""
# # #         device = logits_all.device
# # #         per_cat_correct = torch.zeros(num_classes, device=device)
# # #         per_cat_count = torch.zeros(num_classes, device=device)

# # #         topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # #         for i in torch.unique(labels_all):
# # #             idx = labels_all == i
# # #             if idx.sum() > 0:
# # #                 per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # #                 per_cat_count[i] = idx.sum()

# # #         valid = per_cat_count > 0
# # #         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # #         class_acc = (per_cat_correct[valid] / per_cat_count[valid]).mean()

# # #         return {
# # #             "overall_acc": overall_acc,
# # #             "class_acc": class_acc,
# # #             "top1": topk_acc[0],
# # #             "top3": topk_acc[1],
# # #             "top5": topk_acc[2],
# # #         }
    

# # #     def _merge_features_labels_dist(self, feat_list, label_list):
# # #         """
# # #         Merge variable-length feature/label tensors across DDP ranks.
# # #         Avoids depending on merge_results_dist signature.
# # #         """
# # #         feat_local = torch.cat(feat_list, dim=0).contiguous()
# # #         labels_local = torch.cat(label_list, dim=0).contiguous()

# # #         if (not dist.is_available()) or (not dist.is_initialized()) or self.config.ngpu <= 1:
# # #             return feat_local, labels_local

# # #         world_size = dist.get_world_size()
# # #         device = feat_local.device

# # #         local_n = torch.tensor([feat_local.shape[0]], device=device, dtype=torch.long)
# # #         sizes_t = [torch.zeros_like(local_n) for _ in range(world_size)]
# # #         dist.all_gather(sizes_t, local_n)

# # #         sizes = [int(x.item()) for x in sizes_t]
# # #         max_n = max(sizes)

# # #         feat_dim = feat_local.shape[1]

# # #         if feat_local.shape[0] < max_n:
# # #             pad_n = max_n - feat_local.shape[0]

# # #             feat_pad = torch.zeros(
# # #                 pad_n,
# # #                 feat_dim,
# # #                 device=device,
# # #                 dtype=feat_local.dtype,
# # #             )

# # #             label_pad = torch.zeros(
# # #                 pad_n,
# # #                 device=device,
# # #                 dtype=labels_local.dtype,
# # #             )

# # #             feat_local = torch.cat([feat_local, feat_pad], dim=0)
# # #             labels_local = torch.cat([labels_local, label_pad], dim=0)

# # #         gathered_feats = [torch.zeros_like(feat_local) for _ in range(world_size)]
# # #         gathered_labels = [torch.zeros_like(labels_local) for _ in range(world_size)]

# # #         dist.all_gather(gathered_feats, feat_local)
# # #         dist.all_gather(gathered_labels, labels_local)

# # #         if self.rank == 0:
# # #             feat_all = torch.cat(
# # #                 [gathered_feats[r][:sizes[r]] for r in range(world_size)],
# # #                 dim=0,
# # #             )

# # #             labels_all = torch.cat(
# # #                 [gathered_labels[r][:sizes[r]] for r in range(world_size)],
# # #                 dim=0,
# # #             )

# # #             return feat_all, labels_all

# # #         return None, None


# # #     def test_zero_shot_truncated(self, loader, dataset_name, num_classes, merge_name):
# # #         """
# # #         OpenShape truncated baseline.

# # #         Protocol:
# # #         1) Extract the full OpenShape 3D feature, usually 1280-D.
# # #         2) Merge features across DDP ranks once.
# # #         3) For each d, evaluate F.normalize(feat[:, :d]) @ F.normalize(text[:, :d]).T.

# # #         This does not use MRL heads and does not train anything.
# # #         """
# # #         self.model.eval()
# # #         if self.config.training.use_text_proj:
# # #             self.text_proj.eval()

# # #         clip_text_feat = torch.from_numpy(loader.dataset.clip_cat_feat).to(self.config.device)
# # #         if self.config.training.use_text_proj:
# # #             clip_text_feat = self.text_proj(clip_text_feat)

# # #         feat_all = []
# # #         labels_all = []
# # #         with torch.no_grad():
# # #             for data in tqdm(loader):
# # #                 pred_feat = self._forward_pointcloud(data)
# # #                 labels = data['category'].to(self.config.device)

# # #                 feat_all.append(pred_feat.detach())
# # #                 labels_all.append(labels.detach())

# # #         # feat_all, labels_all = merge_results_dist(
# # #         #     os.path.join(self.config.ckpt_dir, "{}_features_for_truncated".format(merge_name)),
# # #         #     feat_all,
# # #         #     labels_all,
# # #         # )
# # #         feat_all, labels_all = self._merge_features_labels_dist(feat_all, labels_all)


# # #         if self.rank == 0:
# # #             clip_text_feat = clip_text_feat.to(feat_all.device)
# # #             max_dim = min(feat_all.shape[1], clip_text_feat.shape[1])
# # #             dims = [d for d in self._get_truncated_dims() if d <= max_dim]

# # #             if len(dims) == 0:
# # #                 raise ValueError("No valid truncated dimensions. max_dim={} dims={}".format(
# # #                     max_dim, self._get_truncated_dims()))

# # #             logging.info("========== OpenShape truncated: {} ==========".format(dataset_name))
# # #             logging.info("Feature dim: shape={} text={} | eval dims={}".format(
# # #                 tuple(feat_all.shape), tuple(clip_text_feat.shape), dims))

# # #             results = {}
# # #             for d in dims:
# # #                 shape_d = F.normalize(feat_all[:, :d], dim=1)
# # #                 text_d = F.normalize(clip_text_feat[:, :d], dim=1)
# # #                 logits_all = shape_d @ text_d.T

# # #                 stats = self._compute_zero_shot_stats(logits_all, labels_all, num_classes)
# # #                 results[d] = {k: float(v.item()) for k, v in stats.items()}

# # #                 logging.info(
# # #                     "OpenShape truncated | {} | D={}: overall_acc: {:.4f} class_acc: {:.4f} "
# # #                     "top1_acc: {:.4f} top3_acc: {:.4f} top5_acc: {:.4f}".format(
# # #                         dataset_name,
# # #                         d,
# # #                         results[d]["overall_acc"],
# # #                         results[d]["class_acc"],
# # #                         results[d]["top1"],
# # #                         results[d]["top3"],
# # #                         results[d]["top5"],
# # #                     )
# # #                 )

# # #             return results

# # #         return None

    

# # #     def test_modelnet40_truncated(self):
# # #         return self.test_zero_shot_truncated(
# # #             loader=self.modelnet40_loader,
# # #             dataset_name="ModelNet40",
# # #             num_classes=40,
# # #             merge_name="modelnet40_truncated",
# # #         )

# # #     def test_objaverse_lvis_truncated(self):
# # #         return self.test_zero_shot_truncated(
# # #             loader=self.objaverse_lvis_loader,
# # #             dataset_name="ObjaverseLVIS",
# # #             num_classes=1156,
# # #             merge_name="objaverse_lvis_truncated",
# # #         )

# # #     def test_scanobjectnn_truncated(self):
# # #         return self.test_zero_shot_truncated(
# # #             loader=self.scanobjectnn_loader,
# # #             dataset_name="ScanObjectNN",
# # #             num_classes=15,
# # #             merge_name="scanobjectnn_truncated",
# # #         )

# # #     def train(self):
# # #         for epoch in range(self.epoch, self.config.training.max_epoch):
# # #             self.epoch = epoch
# # #             if self.rank == 0:
# # #                 logging.info("Epoch: {}".format(self.epoch))
# # #             self.train_one_epoch()
# # #             if epoch > self.config.training.test_epoch:
# # #                 self.test_objaverse_lvis_truncated()
# # #                 self.test_modelnet40_truncated()
# # #                 self.test_scanobjectnn_truncated()
# # #             if self.rank == 0:
# # #                 self.save_model('latest')
# # #             if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
# # #                 self.save_model('epoch_{}'.format(self.epoch))


# # #     def test_modelnet40(self):
# # #         self.model.eval()
# # #         if self.config.training.use_text_proj:
# # #             self.text_proj.eval()
# # #         clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).cuda()
# # #         if self.config.training.use_text_proj:
# # #             clip_text_feat = self.text_proj(clip_text_feat)
# # #         per_cat_correct = torch.zeros(40).cuda()
# # #         per_cat_count = torch.zeros(40).cuda()
# # #         category2idx = self.modelnet40_loader.dataset.category2idx
# # #         idx2category = {v: k for k, v in category2idx.items()}

# # #         logits_all = []
# # #         labels_all = []
# # #         with torch.no_grad():
# # #             for data in self.modelnet40_loader:
# # #                 if not self.config.model.get("use_dense", False):
# # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # #                                            device=self.config.device, \
# # #                                            quantization_size=self.config.model.voxel_size)
# # #                 else:
# # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])

# # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # #                 labels = data['category'].to(self.config.device)
# # #                 logits_all.append(logits.detach())
# # #                 labels_all.append(labels)

# # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "modelnet40_dir"),
# # #                                                     logits_all, labels_all)

# # #         if self.rank == 0:

# # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # #             for i in range(40):
# # #                 idx = (labels_all == i)
# # #                 if idx.sum() > 0:
# # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # #                     per_cat_count[i] = idx.sum()

# # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # #             per_cat_acc = per_cat_correct / per_cat_count
# # #             # for i in range(40):
# # #             #    print(idx2category[i], per_cat_acc[i])

# # #             if overall_acc > self.best_modelnet40_overall_acc:
# # #                 self.best_modelnet40_overall_acc = overall_acc
# # #                 # self.save_model('best_modelnet40_overall')
# # #             if per_cat_acc.mean() > self.best_modelnet40_class_acc:
# # #                 self.best_modelnet40_class_acc = per_cat_acc.mean()
# # #                 # self.save_model('best_modelnet40_class')

# # #             logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
# # #                                                                                              self.best_modelnet40_overall_acc,
# # #                                                                                              per_cat_acc.mean(),
# # #                                                                                              self.best_modelnet40_class_acc))
# # #             logging.info(
# # #                 'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # #                                                                                     topk_acc[1].item(),
# # #                                                                                     topk_acc[2].item()))
# # #             # wandb.log({"test/epoch": self.epoch,
# # #             #            "test/step": self.step,
# # #             #            "test/ModelNet40_overall_acc": overall_acc,
# # #             #            "test/ModelNet40_class_acc": per_cat_acc.mean(),
# # #             #            "test/top3_acc": topk_acc[1],
# # #             #            "test/top5_acc": topk_acc[2], })

# # #     def test_objaverse_lvis(self):
# # #         self.model.eval()
# # #         if self.config.training.use_text_proj:
# # #             self.text_proj.eval()
# # #         clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).cuda()
# # #         if self.config.training.use_text_proj:
# # #             clip_text_feat = self.text_proj(clip_text_feat)
# # #         per_cat_correct = torch.zeros(1156).cuda()
# # #         per_cat_count = torch.zeros(1156).cuda()
# # #         category2idx = self.objaverse_lvis_loader.dataset.category2idx
# # #         idx2category = {v: k for k, v in category2idx.items()}

# # #         logits_all = []
# # #         labels_all = []
# # #         with torch.no_grad():
# # #             for data in self.objaverse_lvis_loader:
# # #                 if not self.config.model.get("use_dense", False):
# # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # #                                            device=self.config.device, \
# # #                                            quantization_size=self.config.model.voxel_size)
# # #                 else:
# # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # #                 labels = data['category'].to(self.config.device)
# # #                 logits_all.append(logits.detach())
# # #                 labels_all.append(labels)

# # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "objaverse_dir"),
# # #                                                     logits_all, labels_all)

# # #         if self.rank == 0:
# # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # #             # calculate per class accuracy
# # #             for i in torch.unique(labels_all):
# # #                 idx = (labels_all == i)
# # #                 if idx.sum() > 0:
# # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # #                     per_cat_count[i] = idx.sum()

# # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # #             per_cat_acc = per_cat_correct / per_cat_count

# # #             if overall_acc > self.best_lvis_acc:
# # #                 self.best_lvis_acc = overall_acc
# # #                 # self.save_model('best_lvis')

# # #             logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# # #             logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # #                                                                                                 topk_acc[1].item(),
# # #                                                                                                 topk_acc[2].item()))
# # #             # wandb.log({"test_lvis/epoch": self.epoch,
# # #             #            "test_lvis/step": self.step,
# # #             #            "test_lvis/overall_acc": overall_acc,
# # #             #            "test_lvis/class_acc": per_cat_acc.mean(),
# # #             #            "test_lvis/top3_acc": topk_acc[1],
# # #             #            "test_lvis/top5_acc": topk_acc[2], })

# # #     def test_scanobjectnn(self):
# # #         self.model.eval()
# # #         if self.config.training.use_text_proj:
# # #             self.text_proj.eval()
# # #         clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
# # #         if self.config.training.use_text_proj:
# # #             clip_text_feat = self.text_proj(clip_text_feat)
# # #         per_cat_correct = torch.zeros(15).to(self.config.device)
# # #         per_cat_count = torch.zeros(15).to(self.config.device)
# # #         category2idx = self.scanobjectnn_loader.dataset.category2idx
# # #         idx2category = {v: k for k, v in category2idx.items()}

# # #         logits_all = []
# # #         labels_all = []
# # #         with torch.no_grad():
# # #             for data in self.scanobjectnn_loader:
# # #                 if not self.config.model.get("use_dense", False):
# # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # #                                            device=self.config.device, \
# # #                                            quantization_size=self.config.model.voxel_size)
# # #                 else:
# # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # #                 labels = data['category'].to(self.config.device)
# # #                 logits_all.append(logits.detach())
# # #                 labels_all.append(labels)

# # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scanobjectnn_dir"),
# # #                                                     logits_all, labels_all)

# # #         if self.rank == 0:

# # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # #             # calculate per class accuracy
# # #             for i in range(15):
# # #                 idx = (labels_all == i)
# # #                 if idx.sum() > 0:
# # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # #                     per_cat_count[i] = idx.sum()

# # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # #             per_cat_acc = per_cat_correct / per_cat_count

# # #             logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# # #             logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # #                                                                                                topk_acc[1].item(),
# # #                                                                                                topk_acc[2].item()))
# # #             # wandb.log({"test_scanobjectnn/epoch": self.epoch,
# # #             #            "test_scanobjectnn/step": self.step,
# # #             #            "test_scanobjectnn/overall_acc": overall_acc,
# # #             #            "test_scanobjectnn/class_acc": per_cat_acc.mean(),
# # #             #            "test_scanobjectnn/top3_acc": topk_acc[1],
# # #             #            "test_scanobjectnn/top5_acc": topk_acc[2], })


# # #     def test_scannet(self, scannet_loader):
# # #         self.model.eval()
# # #         if self.config.training.use_text_proj:
# # #             self.text_proj.eval()
# # #         clip_text_feat = torch.from_numpy(scannet_loader.dataset.clip_cat_feat).to(self.config.device)
# # #         if self.config.training.use_text_proj:
# # #             clip_text_feat = self.text_proj(clip_text_feat)
# # #         per_cat_correct = torch.zeros(19).to(self.config.device)
# # #         per_cat_count = torch.zeros(19).to(self.config.device)
# # #         category2idx = scannet_loader.dataset.category2idx
# # #         idx2category = {v: k for k, v in category2idx.items()}

# # #         logits_all = []
# # #         labels_all = []
# # #         with torch.no_grad():
# # #             for data in tqdm(scannet_loader):
# # #                 if not self.config.model.get("use_dense", False):
# # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # #                                            device=self.config.device, \
# # #                                            quantization_size=self.config.model.voxel_size)
# # #                 else:
# # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # #                 labels = data['category'].to(self.config.device)
# # #                 logits_all.append(logits.detach())
# # #                 labels_all.append(labels)

# # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scannet_dir"),
# # #                                                     logits_all, labels_all)
# # #         if self.rank == 0:

# # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # #             # calculate per class accuracy
# # #             for i in range(19):
# # #                 idx = (logits_all == i)
# # #                 if idx.sum() > 0:
# # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # #                     per_cat_count[i] = idx.sum()

# # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # #             per_cat_acc = per_cat_correct / per_cat_count


# # #             logging.info('Test Scannet: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# # #             logging.info('Test Scannet: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # #                                                                                            topk_acc[1].item(),
# # #                                                                                                topk_acc[2].item()))

# # # # import logging
# # # # import os

# # # # import numpy as np
# # # # import torch
# # # # import torch.distributed.nn
# # # # import torch.nn.functional as F
# # # # import wandb
# # # # import torch.distributed as dist
# # # # from trainers.trainer_utils import merge_results_dist
# # # # from tqdm import tqdm
# # # # from trainers.trainer_utils import merge_two_branch_results_dist
# # # # from trainers.trainer_utils import merge_results_dist



# # # # class TrainerOpenShape(object):
# # # #     def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, train_loader, \
# # # #                  modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None, clip_adapter=None):
# # # #         self.rank = rank
# # # #         self.config = config
# # # #         self.model = model
# # # #         self.logit_scale = logit_scale
# # # #         self.image_proj = image_proj
# # # #         self.text_proj = text_proj
# # # #         self.optimizer = optimizer
# # # #         self.scheduler = scheduler
# # # #         self.train_loader = train_loader
# # # #         self.modelnet40_loader = modelnet40_loader
# # # #         self.objaverse_lvis_loader = objaverse_lvis_loader
# # # #         self.scanobjectnn_loader = scanobjectnn_loader
# # # #         self.epoch = 0
# # # #         self.step = 0
# # # #         self.clip_adapter = clip_adapter
# # # #         self.best_img_contras_acc = 0
# # # #         self.best_text_contras_acc = 0
# # # #         self.best_modelnet40_overall_acc = 0
# # # #         self.best_modelnet40_class_acc = 0
# # # #         self.best_lvis_acc = 0
# # # #         self.config.ngpu = dist.get_world_size()

# # # #     def _get_module(self, module):
# # # #         """Return the wrapped module when using DataParallel/DDP."""
# # # #         return module.module if hasattr(module, "module") else module

# # # #     def _clean_ddp_state_dict(self, state_dict):
# # # #         """Remove the `module.` prefix from checkpoints saved with DDP."""
# # # #         if state_dict is None:
# # # #             return None
# # # #         return {k[len("module."):] if k.startswith("module.") else k: v
# # # #                 for k, v in state_dict.items()}

# # # #     def _load_state_dict_clean(self, module, state_dict, strict=True, name="module"):
# # # #         """Load a state dict robustly, handling DDP/non-DDP checkpoints."""
# # # #         state_dict = self._clean_ddp_state_dict(state_dict)
# # # #         msg = self._get_module(module).load_state_dict(state_dict, strict=strict)
# # # #         logging.info("Loaded {} state_dict: {}".format(name, msg))
# # # #         return msg

# # # #     def load_from_checkpoint(self, path, resume=True, strict=True):
# # # #         checkpoint = torch.load(path, map_location="cpu")

# # # #         self._load_state_dict_clean(self.model, checkpoint['state_dict'], strict=strict, name="model")

# # # #         if 'logit_scale' in checkpoint and checkpoint['logit_scale'] is not None:
# # # #             self._load_state_dict_clean(self.logit_scale, checkpoint['logit_scale'], strict=strict, name="logit_scale")

# # # #         if self.config.training.use_text_proj and 'text_proj' in checkpoint and checkpoint['text_proj'] is not None:
# # # #             self._load_state_dict_clean(self.text_proj, checkpoint['text_proj'], strict=strict, name="text_proj")

# # # #         if self.config.training.use_image_proj and 'image_proj' in checkpoint and checkpoint['image_proj'] is not None:
# # # #             self._load_state_dict_clean(self.image_proj, checkpoint['image_proj'], strict=strict, name="image_proj")

# # # #         # Use resume=False for evaluation-only baselines such as OpenShape truncated.
# # # #         if resume:
# # # #             if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
# # # #                 self.optimizer.load_state_dict(checkpoint['optimizer'])
# # # #             if self.config.training.use_openclip_optimizer_scheduler == False:
# # # #                 if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
# # # #                     self.scheduler.load_state_dict(checkpoint['scheduler'])
# # # #             self.epoch = checkpoint.get('epoch', 0)
# # # #             self.step = checkpoint.get('step', 0)
# # # #         else:
# # # #             self.epoch = 0
# # # #             self.step = 0

# # # #         logging.info("Loaded checkpoint from {}".format(path))
# # # #         logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

# # # #     def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
# # # #         if self.config.ngpu > 1:
# # # #             feat1 = F.normalize(feat1, dim=1)
# # # #             feat2 = F.normalize(feat2, dim=1)
# # # #             all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
# # # #             all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
# # # #             logits = logit_scale * all_feat1 @ all_feat2.T
# # # #         else:
# # # #             logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
# # # #         if mask is not None:
# # # #             logits = logits * mask
# # # #         labels = torch.arange(logits.shape[0]).to(self.config.device)
# # # #         accuracy = (logits.argmax(dim=1) == labels).float().mean()
# # # #         loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
# # # #         return loss, accuracy

# # # #     def train_one_epoch(self):
# # # #         self.model.train()
# # # #         if self.config.training.use_text_proj: #False
# # # #             self.text_proj.train()
# # # #         if self.config.training.use_image_proj: #False
# # # #             self.image_proj.train()

# # # #         text_contras_acc_list = []
# # # #         img_contras_acc_list = []
# # # #         if self.config.training.use_mask:
# # # #             k = self.config.dataset.negative_sample_num
# # # #             s = self.config.dataset.train_batch_size
# # # #             mask1 = np.eye(k * s).astype(np.bool)
# # # #             mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
# # # #             mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

# # # #         for data in tqdm(self.train_loader):
# # # #             self.step += 1
# # # #             self.optimizer.zero_grad()
# # # #             loss = 0
# # # #             if not self.config.model.get("use_dense", False):
# # # #                 pred_feat = self.model(data['xyz'], data['features'], \
# # # #                                        device=self.config.device, \
# # # #                                        quantization_size=self.config.model.voxel_size)
# # # #             else:
# # # #                 pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # # #             logit_scale = self.logit_scale(None)
# # # #             idx = data['has_text_idx']

# # # #             text_feat = torch.vstack(data['text_feat']).to(self.config.device)
# # # #             img_feat = torch.vstack(data['img_feat']).to(self.config.device)

# # # #             if self.config.training.use_mask:
# # # #                 img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
# # # #                 mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
# # # #                 mask = torch.logical_or(mask, mask_other).detach()
# # # #             else:
# # # #                 mask = None

# # # #             if self.clip_adapter is not None:
# # # #                 with torch.no_grad():
# # # #                     img_feat = self.clip_adapter(img_feat)


# # # #             if self.config.training.use_image_proj:
# # # #                 img_feat = self.image_proj(img_feat)
# # # #             img_contras_loss, img_contras_acc = self.contras_loss(pred_feat, img_feat, logit_scale=logit_scale,
# # # #                                                                   mask=mask)

# # # #             loss += img_contras_loss * self.config.training.lambda_img_contras


# # # #             if len(idx) > 0:
# # # #                 if self.config.training.use_text_proj:
# # # #                     text_feat = self.text_proj(text_feat)
# # # #                 text_contras_loss, text_contras_acc = self.contras_loss(pred_feat[idx], text_feat,
# # # #                                                                         logit_scale=logit_scale, mask=mask)

# # # #                 loss += text_contras_loss * self.config.training.lambda_text_contras



# # # #             text_contras_acc_list.append(text_contras_acc.item())
# # # #             img_contras_acc_list.append(img_contras_acc.item())
# # # #             loss.backward()
# # # #             self.optimizer.step()
# # # #             if self.config.training.use_openclip_optimizer_scheduler:
# # # #                 self.scheduler(self.step)
# # # #             else:
# # # #                 self.scheduler.step()

# # # #         if self.rank == 0:
# # # #             logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}' \
# # # #                          .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
# # # #                                  np.mean(img_contras_acc_list)))

# # # #     def save_model(self, name):
# # # #         torch.save({
# # # #             "state_dict": self.model.state_dict(),
# # # #             "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
# # # #             "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
# # # #             "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
# # # #             "optimizer": self.optimizer.state_dict(),
# # # #             "scheduler": self.scheduler.state_dict() if self.config.training.use_openclip_optimizer_scheduler == False else None,
# # # #             "epoch": self.epoch,
# # # #             "step": self.step,
# # # #             "best_img_contras_acc": self.best_img_contras_acc,
# # # #             "best_text_contras_acc": self.best_text_contras_acc,
# # # #             "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
# # # #             "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
# # # #             "best_lvis_acc": self.best_lvis_acc,
# # # #         }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

# # # #     def accuracy(self, output, target, topk=(1,)):
# # # #         """Computes the accuracy over the k top predictions for the specified values of k"""
# # # #         with torch.no_grad():
# # # #             maxk = max(topk)
# # # #             batch_size = target.size(0)

# # # #             _, pred = output.topk(maxk, 1, True, True)
# # # #             pred = pred.t()
# # # #             correct = pred.eq(target.reshape(1, -1).expand_as(pred))

# # # #             res = []
# # # #             for k in topk:
# # # #                 correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
# # # #                 res.append(correct_k.mul_(100.0 / batch_size))
# # # #             return res, correct

# # # #     def _cfg_get(self, obj, key, default=None):
# # # #         """Works with dict-like configs and EasyDict/OmegaConf-style configs."""
# # # #         if obj is None:
# # # #             return default
# # # #         if hasattr(obj, "get"):
# # # #             return obj.get(key, default)
# # # #         return getattr(obj, key, default)

# # # #     def _get_truncated_dims(self):
# # # #         """Dimensions used for the OpenShape truncated baseline."""
# # # #         default_dims = [10, 20, 40, 80, 160, 320, 640, 1280]

# # # #         # Priority 1: config.eval.truncated_dims
# # # #         eval_cfg = self._cfg_get(self.config, "eval", None)
# # # #         dims = self._cfg_get(eval_cfg, "truncated_dims", None)

# # # #         # Priority 2: config.model.truncated_dims
# # # #         if dims is None:
# # # #             dims = self._cfg_get(self.config.model, "truncated_dims", None)

# # # #         # Priority 3: config.model.nesting_list, useful if you want the exact MRL scales
# # # #         if dims is None:
# # # #             dims = self._cfg_get(self.config.model, "nesting_list", None)

# # # #         if dims is None:
# # # #             dims = default_dims

# # # #         return [int(d) for d in dims]

# # # #     def _forward_pointcloud(self, data):
# # # #         """Forward pass for the 3D encoder, shared by normal and truncated evaluation."""
# # # #         if not self.config.model.get("use_dense", False):
# # # #             return self.model(data['xyz'], data['features'],
# # # #                               device=self.config.device,
# # # #                               quantization_size=self.config.model.voxel_size)
# # # #         return self.model(data['xyz_dense'], data['features_dense'])

# # # #     def _compute_zero_shot_stats(self, logits_all, labels_all, num_classes):
# # # #         """Compute overall acc, mean class acc, and top-k accuracy."""
# # # #         device = logits_all.device
# # # #         per_cat_correct = torch.zeros(num_classes, device=device)
# # # #         per_cat_count = torch.zeros(num_classes, device=device)

# # # #         topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # # #         for i in torch.unique(labels_all):
# # # #             idx = labels_all == i
# # # #             if idx.sum() > 0:
# # # #                 per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # # #                 per_cat_count[i] = idx.sum()

# # # #         valid = per_cat_count > 0
# # # #         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # # #         class_acc = (per_cat_correct[valid] / per_cat_count[valid]).mean()

# # # #         return {
# # # #             "overall_acc": overall_acc,
# # # #             "class_acc": class_acc,
# # # #             "top1": topk_acc[0],
# # # #             "top3": topk_acc[1],
# # # #             "top5": topk_acc[2],
# # # #         }

# # # #     def test_zero_shot_truncated(self, loader, dataset_name, num_classes, merge_name):
# # # #         """
# # # #         OpenShape truncated baseline.

# # # #         Protocol:
# # # #         1) Extract the full OpenShape 3D feature, usually 1280-D.
# # # #         2) Merge features across DDP ranks once.
# # # #         3) For each d, evaluate F.normalize(feat[:, :d]) @ F.normalize(text[:, :d]).T.

# # # #         This does not use MRL heads and does not train anything.
# # # #         """
# # # #         self.model.eval()
# # # #         if self.config.training.use_text_proj:
# # # #             self.text_proj.eval()

# # # #         clip_text_feat = torch.from_numpy(loader.dataset.clip_cat_feat).to(self.config.device)
# # # #         if self.config.training.use_text_proj:
# # # #             clip_text_feat = self.text_proj(clip_text_feat)

# # # #         feat_all = []
# # # #         labels_all = []
# # # #         with torch.no_grad():
# # # #             for data in tqdm(loader):
# # # #                 pred_feat = self._forward_pointcloud(data)
# # # #                 labels = data['category'].to(self.config.device)

# # # #                 feat_all.append(pred_feat.detach())
# # # #                 labels_all.append(labels.detach())

# # # #         feat_all, labels_all = merge_results_dist(
# # # #             os.path.join(self.config.ckpt_dir, "{}_features_for_truncated".format(merge_name)),
# # # #             feat_all,
# # # #             labels_all,
# # # #         )

# # # #         if self.rank == 0:
# # # #             clip_text_feat = clip_text_feat.to(feat_all.device)
# # # #             max_dim = min(feat_all.shape[1], clip_text_feat.shape[1])
# # # #             dims = [d for d in self._get_truncated_dims() if d <= max_dim]

# # # #             if len(dims) == 0:
# # # #                 raise ValueError("No valid truncated dimensions. max_dim={} dims={}".format(
# # # #                     max_dim, self._get_truncated_dims()))

# # # #             logging.info("========== OpenShape truncated: {} ==========".format(dataset_name))
# # # #             logging.info("Feature dim: shape={} text={} | eval dims={}".format(
# # # #                 tuple(feat_all.shape), tuple(clip_text_feat.shape), dims))

# # # #             results = {}
# # # #             for d in dims:
# # # #                 shape_d = F.normalize(feat_all[:, :d], dim=1)
# # # #                 text_d = F.normalize(clip_text_feat[:, :d], dim=1)
# # # #                 logits_all = shape_d @ text_d.T

# # # #                 stats = self._compute_zero_shot_stats(logits_all, labels_all, num_classes)
# # # #                 results[d] = {k: float(v.item()) for k, v in stats.items()}

# # # #                 logging.info(
# # # #                     "OpenShape truncated | {} | D={}: overall_acc: {:.4f} class_acc: {:.4f} "
# # # #                     "top1_acc: {:.4f} top3_acc: {:.4f} top5_acc: {:.4f}".format(
# # # #                         dataset_name,
# # # #                         d,
# # # #                         results[d]["overall_acc"],
# # # #                         results[d]["class_acc"],
# # # #                         results[d]["top1"],
# # # #                         results[d]["top3"],
# # # #                         results[d]["top5"],
# # # #                     )
# # # #                 )

# # # #             return results

# # # #         return None

# # # #     def test_modelnet40_truncated(self):
# # # #         return self.test_zero_shot_truncated(
# # # #             loader=self.modelnet40_loader,
# # # #             dataset_name="ModelNet40",
# # # #             num_classes=40,
# # # #             merge_name="modelnet40_truncated",
# # # #         )

# # # #     def test_objaverse_lvis_truncated(self):
# # # #         return self.test_zero_shot_truncated(
# # # #             loader=self.objaverse_lvis_loader,
# # # #             dataset_name="ObjaverseLVIS",
# # # #             num_classes=1156,
# # # #             merge_name="objaverse_lvis_truncated",
# # # #         )

# # # #     def test_scanobjectnn_truncated(self):
# # # #         return self.test_zero_shot_truncated(
# # # #             loader=self.scanobjectnn_loader,
# # # #             dataset_name="ScanObjectNN",
# # # #             num_classes=15,
# # # #             merge_name="scanobjectnn_truncated",
# # # #         )

# # # #     def train(self):
# # # #         for epoch in range(self.epoch, self.config.training.max_epoch):
# # # #             self.epoch = epoch
# # # #             if self.rank == 0:
# # # #                 logging.info("Epoch: {}".format(self.epoch))
# # # #             self.train_one_epoch()
# # # #             if epoch > self.config.training.test_epoch:
# # # #                 self.test_objaverse_lvis()
# # # #                 # self.test_modelnet40()
# # # #                 self.test_objaverse_lvis()
# # # #             if self.rank == 0:
# # # #                 self.save_model('latest')
# # # #             if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
# # # #                 self.save_model('epoch_{}'.format(self.epoch))


# # # #     def test_modelnet40(self):
# # # #         self.model.eval()
# # # #         if self.config.training.use_text_proj:
# # # #             self.text_proj.eval()
# # # #         clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).cuda()
# # # #         if self.config.training.use_text_proj:
# # # #             clip_text_feat = self.text_proj(clip_text_feat)
# # # #         per_cat_correct = torch.zeros(40).cuda()
# # # #         per_cat_count = torch.zeros(40).cuda()
# # # #         category2idx = self.modelnet40_loader.dataset.category2idx
# # # #         idx2category = {v: k for k, v in category2idx.items()}

# # # #         logits_all = []
# # # #         labels_all = []
# # # #         with torch.no_grad():
# # # #             for data in self.modelnet40_loader:
# # # #                 if not self.config.model.get("use_dense", False):
# # # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # # #                                            device=self.config.device, \
# # # #                                            quantization_size=self.config.model.voxel_size)
# # # #                 else:
# # # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])

# # # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # # #                 labels = data['category'].to(self.config.device)
# # # #                 logits_all.append(logits.detach())
# # # #                 labels_all.append(labels)

# # # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "modelnet40_dir"),
# # # #                                                     logits_all, labels_all)

# # # #         if self.rank == 0:

# # # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # # #             for i in range(40):
# # # #                 idx = (labels_all == i)
# # # #                 if idx.sum() > 0:
# # # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # # #                     per_cat_count[i] = idx.sum()

# # # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # # #             per_cat_acc = per_cat_correct / per_cat_count
# # # #             # for i in range(40):
# # # #             #    print(idx2category[i], per_cat_acc[i])

# # # #             if overall_acc > self.best_modelnet40_overall_acc:
# # # #                 self.best_modelnet40_overall_acc = overall_acc
# # # #                 # self.save_model('best_modelnet40_overall')
# # # #             if per_cat_acc.mean() > self.best_modelnet40_class_acc:
# # # #                 self.best_modelnet40_class_acc = per_cat_acc.mean()
# # # #                 # self.save_model('best_modelnet40_class')

# # # #             logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
# # # #                                                                                              self.best_modelnet40_overall_acc,
# # # #                                                                                              per_cat_acc.mean(),
# # # #                                                                                              self.best_modelnet40_class_acc))
# # # #             logging.info(
# # # #                 'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # # #                                                                                     topk_acc[1].item(),
# # # #                                                                                     topk_acc[2].item()))
# # # #             # wandb.log({"test/epoch": self.epoch,
# # # #             #            "test/step": self.step,
# # # #             #            "test/ModelNet40_overall_acc": overall_acc,
# # # #             #            "test/ModelNet40_class_acc": per_cat_acc.mean(),
# # # #             #            "test/top3_acc": topk_acc[1],
# # # #             #            "test/top5_acc": topk_acc[2], })

# # # #     def test_objaverse_lvis(self):
# # # #         self.model.eval()
# # # #         if self.config.training.use_text_proj:
# # # #             self.text_proj.eval()
# # # #         clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).cuda()
# # # #         if self.config.training.use_text_proj:
# # # #             clip_text_feat = self.text_proj(clip_text_feat)
# # # #         per_cat_correct = torch.zeros(1156).cuda()
# # # #         per_cat_count = torch.zeros(1156).cuda()
# # # #         category2idx = self.objaverse_lvis_loader.dataset.category2idx
# # # #         idx2category = {v: k for k, v in category2idx.items()}

# # # #         logits_all = []
# # # #         labels_all = []
# # # #         with torch.no_grad():
# # # #             for data in self.objaverse_lvis_loader:
# # # #                 if not self.config.model.get("use_dense", False):
# # # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # # #                                            device=self.config.device, \
# # # #                                            quantization_size=self.config.model.voxel_size)
# # # #                 else:
# # # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # # #                 labels = data['category'].to(self.config.device)
# # # #                 logits_all.append(logits.detach())
# # # #                 labels_all.append(labels)

# # # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "objaverse_dir"),
# # # #                                                     logits_all, labels_all)

# # # #         if self.rank == 0:
# # # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # # #             # calculate per class accuracy
# # # #             for i in torch.unique(labels_all):
# # # #                 idx = (labels_all == i)
# # # #                 if idx.sum() > 0:
# # # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # # #                     per_cat_count[i] = idx.sum()

# # # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # # #             per_cat_acc = per_cat_correct / per_cat_count

# # # #             if overall_acc > self.best_lvis_acc:
# # # #                 self.best_lvis_acc = overall_acc
# # # #                 # self.save_model('best_lvis')

# # # #             logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# # # #             logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # # #                                                                                                 topk_acc[1].item(),
# # # #                                                                                                 topk_acc[2].item()))
# # # #             # wandb.log({"test_lvis/epoch": self.epoch,
# # # #             #            "test_lvis/step": self.step,
# # # #             #            "test_lvis/overall_acc": overall_acc,
# # # #             #            "test_lvis/class_acc": per_cat_acc.mean(),
# # # #             #            "test_lvis/top3_acc": topk_acc[1],
# # # #             #            "test_lvis/top5_acc": topk_acc[2], })

# # # #     def test_scanobjectnn(self):
# # # #         self.model.eval()
# # # #         if self.config.training.use_text_proj:
# # # #             self.text_proj.eval()
# # # #         clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
# # # #         if self.config.training.use_text_proj:
# # # #             clip_text_feat = self.text_proj(clip_text_feat)
# # # #         per_cat_correct = torch.zeros(15).to(self.config.device)
# # # #         per_cat_count = torch.zeros(15).to(self.config.device)
# # # #         category2idx = self.scanobjectnn_loader.dataset.category2idx
# # # #         idx2category = {v: k for k, v in category2idx.items()}

# # # #         logits_all = []
# # # #         labels_all = []
# # # #         with torch.no_grad():
# # # #             for data in self.scanobjectnn_loader:
# # # #                 if not self.config.model.get("use_dense", False):
# # # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # # #                                            device=self.config.device, \
# # # #                                            quantization_size=self.config.model.voxel_size)
# # # #                 else:
# # # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # # #                 labels = data['category'].to(self.config.device)
# # # #                 logits_all.append(logits.detach())
# # # #                 labels_all.append(labels)

# # # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scanobjectnn_dir"),
# # # #                                                     logits_all, labels_all)

# # # #         if self.rank == 0:

# # # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # # #             # calculate per class accuracy
# # # #             for i in range(15):
# # # #                 idx = (labels_all == i)
# # # #                 if idx.sum() > 0:
# # # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # # #                     per_cat_count[i] = idx.sum()

# # # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # # #             per_cat_acc = per_cat_correct / per_cat_count

# # # #             logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# # # #             logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # # #                                                                                                topk_acc[1].item(),
# # # #                                                                                                topk_acc[2].item()))
# # # #             # wandb.log({"test_scanobjectnn/epoch": self.epoch,
# # # #             #            "test_scanobjectnn/step": self.step,
# # # #             #            "test_scanobjectnn/overall_acc": overall_acc,
# # # #             #            "test_scanobjectnn/class_acc": per_cat_acc.mean(),
# # # #             #            "test_scanobjectnn/top3_acc": topk_acc[1],
# # # #             #            "test_scanobjectnn/top5_acc": topk_acc[2], })


# # # #     def test_scannet(self, scannet_loader):
# # # #         self.model.eval()
# # # #         if self.config.training.use_text_proj:
# # # #             self.text_proj.eval()
# # # #         clip_text_feat = torch.from_numpy(scannet_loader.dataset.clip_cat_feat).to(self.config.device)
# # # #         if self.config.training.use_text_proj:
# # # #             clip_text_feat = self.text_proj(clip_text_feat)
# # # #         per_cat_correct = torch.zeros(19).to(self.config.device)
# # # #         per_cat_count = torch.zeros(19).to(self.config.device)
# # # #         category2idx = scannet_loader.dataset.category2idx
# # # #         idx2category = {v: k for k, v in category2idx.items()}

# # # #         logits_all = []
# # # #         labels_all = []
# # # #         with torch.no_grad():
# # # #             for data in tqdm(scannet_loader):
# # # #                 if not self.config.model.get("use_dense", False):
# # # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # # #                                            device=self.config.device, \
# # # #                                            quantization_size=self.config.model.voxel_size)
# # # #                 else:
# # # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # # #                 labels = data['category'].to(self.config.device)
# # # #                 logits_all.append(logits.detach())
# # # #                 labels_all.append(labels)

# # # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scannet_dir"),
# # # #                                                     logits_all, labels_all)
# # # #         if self.rank == 0:

# # # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # # #             # calculate per class accuracy
# # # #             for i in range(19):
# # # #                 idx = (logits_all == i)
# # # #                 if idx.sum() > 0:
# # # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # # #                     per_cat_count[i] = idx.sum()

# # # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # # #             per_cat_acc = per_cat_correct / per_cat_count


# # # #             logging.info('Test Scannet: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# # # #             logging.info('Test Scannet: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # # #                                                                                            topk_acc[1].item(),
# # # #                                                                                                topk_acc[2].item()))


# # # # # import logging
# # # # # import os

# # # # # import numpy as np
# # # # # import torch
# # # # # import torch.distributed.nn
# # # # # import torch.nn.functional as F
# # # # # import wandb
# # # # # import torch.distributed as dist
# # # # # from trainers.trainer_utils import merge_results_dist
# # # # # from tqdm import tqdm
# # # # # from trainers.trainer_utils import merge_two_branch_results_dist


# # # # # class TrainerOpenShape(object):
# # # # #     def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, optimizer, scheduler, train_loader, \
# # # # #                  modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None, clip_adapter=None):
# # # # #         self.rank = rank
# # # # #         self.config = config
# # # # #         self.model = model
# # # # #         self.logit_scale = logit_scale
# # # # #         self.image_proj = image_proj
# # # # #         self.text_proj = text_proj
# # # # #         self.optimizer = optimizer
# # # # #         self.scheduler = scheduler
# # # # #         self.train_loader = train_loader
# # # # #         self.modelnet40_loader = modelnet40_loader
# # # # #         self.objaverse_lvis_loader = objaverse_lvis_loader
# # # # #         self.scanobjectnn_loader = scanobjectnn_loader
# # # # #         self.epoch = 0
# # # # #         self.step = 0
# # # # #         self.clip_adapter = clip_adapter
# # # # #         self.best_img_contras_acc = 0
# # # # #         self.best_text_contras_acc = 0
# # # # #         self.best_modelnet40_overall_acc = 0
# # # # #         self.best_modelnet40_class_acc = 0
# # # # #         self.best_lvis_acc = 0
# # # # #         self.config.ngpu = dist.get_world_size()

# # # # #     def load_from_checkpoint(self, path):
# # # # #         checkpoint = torch.load(path)
# # # # #         self.model.load_state_dict(checkpoint['state_dict'])
# # # # #         self.logit_scale.load_state_dict(checkpoint['logit_scale'])  # module.logit_scale = checkpoint['logit_scale']
# # # # #         self.optimizer.load_state_dict(checkpoint['optimizer'])
# # # # #         if self.config.training.use_openclip_optimizer_scheduler == False:
# # # # #             self.scheduler.load_state_dict(checkpoint['scheduler'])
# # # # #         self.epoch = checkpoint['epoch']
# # # # #         self.step = checkpoint['step']

# # # # #         logging.info("Loaded checkpoint from {}".format(path))
# # # # #         logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

# # # # #     def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
# # # # #         if self.config.ngpu > 1:
# # # # #             feat1 = F.normalize(feat1, dim=1)
# # # # #             feat2 = F.normalize(feat2, dim=1)
# # # # #             all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
# # # # #             all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
# # # # #             logits = logit_scale * all_feat1 @ all_feat2.T
# # # # #         else:
# # # # #             logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
# # # # #         if mask is not None:
# # # # #             logits = logits * mask
# # # # #         labels = torch.arange(logits.shape[0]).to(self.config.device)
# # # # #         accuracy = (logits.argmax(dim=1) == labels).float().mean()
# # # # #         loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
# # # # #         return loss, accuracy

# # # # #     def train_one_epoch(self):
# # # # #         self.model.train()
# # # # #         if self.config.training.use_text_proj: #False
# # # # #             self.text_proj.train()
# # # # #         if self.config.training.use_image_proj: #False
# # # # #             self.image_proj.train()

# # # # #         text_contras_acc_list = []
# # # # #         img_contras_acc_list = []
# # # # #         if self.config.training.use_mask:
# # # # #             k = self.config.dataset.negative_sample_num
# # # # #             s = self.config.dataset.train_batch_size
# # # # #             mask1 = np.eye(k * s).astype(np.bool)
# # # # #             mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
# # # # #             mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

# # # # #         for data in tqdm(self.train_loader):
# # # # #             self.step += 1
# # # # #             self.optimizer.zero_grad()
# # # # #             loss = 0
# # # # #             if not self.config.model.get("use_dense", False):
# # # # #                 pred_feat = self.model(data['xyz'], data['features'], \
# # # # #                                        device=self.config.device, \
# # # # #                                        quantization_size=self.config.model.voxel_size)
# # # # #             else:
# # # # #                 pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # # # #             logit_scale = self.logit_scale(None)
# # # # #             idx = data['has_text_idx']

# # # # #             text_feat = torch.vstack(data['text_feat']).to(self.config.device)
# # # # #             img_feat = torch.vstack(data['img_feat']).to(self.config.device)

# # # # #             if self.config.training.use_mask:
# # # # #                 img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
# # # # #                 mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
# # # # #                 mask = torch.logical_or(mask, mask_other).detach()
# # # # #             else:
# # # # #                 mask = None

# # # # #             if self.clip_adapter is not None:
# # # # #                 with torch.no_grad():
# # # # #                     img_feat = self.clip_adapter(img_feat)


# # # # #             if self.config.training.use_image_proj:
# # # # #                 img_feat = self.image_proj(img_feat)
# # # # #             img_contras_loss, img_contras_acc = self.contras_loss(pred_feat, img_feat, logit_scale=logit_scale,
# # # # #                                                                   mask=mask)

# # # # #             loss += img_contras_loss * self.config.training.lambda_img_contras


# # # # #             if len(idx) > 0:
# # # # #                 if self.config.training.use_text_proj:
# # # # #                     text_feat = self.text_proj(text_feat)
# # # # #                 text_contras_loss, text_contras_acc = self.contras_loss(pred_feat[idx], text_feat,
# # # # #                                                                         logit_scale=logit_scale, mask=mask)

# # # # #                 loss += text_contras_loss * self.config.training.lambda_text_contras



# # # # #             text_contras_acc_list.append(text_contras_acc.item())
# # # # #             img_contras_acc_list.append(img_contras_acc.item())
# # # # #             loss.backward()
# # # # #             self.optimizer.step()
# # # # #             if self.config.training.use_openclip_optimizer_scheduler:
# # # # #                 self.scheduler(self.step)
# # # # #             else:
# # # # #                 self.scheduler.step()

# # # # #         if self.rank == 0:
# # # # #             logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}' \
# # # # #                          .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
# # # # #                                  np.mean(img_contras_acc_list)))

# # # # #     def save_model(self, name):
# # # # #         torch.save({
# # # # #             "state_dict": self.model.state_dict(),
# # # # #             "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
# # # # #             "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
# # # # #             "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
# # # # #             "optimizer": self.optimizer.state_dict(),
# # # # #             "scheduler": self.scheduler.state_dict() if self.config.training.use_openclip_optimizer_scheduler == False else None,
# # # # #             "epoch": self.epoch,
# # # # #             "step": self.step,
# # # # #             "best_img_contras_acc": self.best_img_contras_acc,
# # # # #             "best_text_contras_acc": self.best_text_contras_acc,
# # # # #             "best_modelnet40_overall_acc": self.best_modelnet40_overall_acc,
# # # # #             "best_modelnet40_class_acc": self.best_modelnet40_class_acc,
# # # # #             "best_lvis_acc": self.best_lvis_acc,
# # # # #         }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

# # # # #     def accuracy(self, output, target, topk=(1,)):
# # # # #         """Computes the accuracy over the k top predictions for the specified values of k"""
# # # # #         with torch.no_grad():
# # # # #             maxk = max(topk)
# # # # #             batch_size = target.size(0)

# # # # #             _, pred = output.topk(maxk, 1, True, True)
# # # # #             pred = pred.t()
# # # # #             correct = pred.eq(target.reshape(1, -1).expand_as(pred))

# # # # #             res = []
# # # # #             for k in topk:
# # # # #                 correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
# # # # #                 res.append(correct_k.mul_(100.0 / batch_size))
# # # # #             return res, correct

# # # # #     def train(self):
# # # # #         for epoch in range(self.epoch, self.config.training.max_epoch):
# # # # #             self.epoch = epoch
# # # # #             if self.rank == 0:
# # # # #                 logging.info("Epoch: {}".format(self.epoch))
# # # # #             self.train_one_epoch()
# # # # #             if epoch > self.config.training.test_epoch:
# # # # #                 self.test_objaverse_lvis()
# # # # #                 # self.test_modelnet40()
# # # # #                 self.test_objaverse_lvis()
# # # # #             if self.rank == 0:
# # # # #                 self.save_model('latest')
# # # # #             if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
# # # # #                 self.save_model('epoch_{}'.format(self.epoch))


# # # # #     def test_modelnet40(self):
# # # # #         self.model.eval()
# # # # #         if self.config.training.use_text_proj:
# # # # #             self.text_proj.eval()
# # # # #         clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).cuda()
# # # # #         if self.config.training.use_text_proj:
# # # # #             clip_text_feat = self.text_proj(clip_text_feat)
# # # # #         per_cat_correct = torch.zeros(40).cuda()
# # # # #         per_cat_count = torch.zeros(40).cuda()
# # # # #         category2idx = self.modelnet40_loader.dataset.category2idx
# # # # #         idx2category = {v: k for k, v in category2idx.items()}

# # # # #         logits_all = []
# # # # #         labels_all = []
# # # # #         with torch.no_grad():
# # # # #             for data in self.modelnet40_loader:
# # # # #                 if not self.config.model.get("use_dense", False):
# # # # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # # # #                                            device=self.config.device, \
# # # # #                                            quantization_size=self.config.model.voxel_size)
# # # # #                 else:
# # # # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])

# # # # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # # # #                 labels = data['category'].to(self.config.device)
# # # # #                 logits_all.append(logits.detach())
# # # # #                 labels_all.append(labels)

# # # # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "modelnet40_dir"),
# # # # #                                                     logits_all, labels_all)

# # # # #         if self.rank == 0:

# # # # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # # # #             for i in range(40):
# # # # #                 idx = (labels_all == i)
# # # # #                 if idx.sum() > 0:
# # # # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # # # #                     per_cat_count[i] = idx.sum()

# # # # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # # # #             per_cat_acc = per_cat_correct / per_cat_count
# # # # #             # for i in range(40):
# # # # #             #    print(idx2category[i], per_cat_acc[i])

# # # # #             if overall_acc > self.best_modelnet40_overall_acc:
# # # # #                 self.best_modelnet40_overall_acc = overall_acc
# # # # #                 # self.save_model('best_modelnet40_overall')
# # # # #             if per_cat_acc.mean() > self.best_modelnet40_class_acc:
# # # # #                 self.best_modelnet40_class_acc = per_cat_acc.mean()
# # # # #                 # self.save_model('best_modelnet40_class')

# # # # #             logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
# # # # #                                                                                              self.best_modelnet40_overall_acc,
# # # # #                                                                                              per_cat_acc.mean(),
# # # # #                                                                                              self.best_modelnet40_class_acc))
# # # # #             logging.info(
# # # # #                 'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # # # #                                                                                     topk_acc[1].item(),
# # # # #                                                                                     topk_acc[2].item()))
# # # # #             # wandb.log({"test/epoch": self.epoch,
# # # # #             #            "test/step": self.step,
# # # # #             #            "test/ModelNet40_overall_acc": overall_acc,
# # # # #             #            "test/ModelNet40_class_acc": per_cat_acc.mean(),
# # # # #             #            "test/top3_acc": topk_acc[1],
# # # # #             #            "test/top5_acc": topk_acc[2], })

# # # # #     def test_objaverse_lvis(self):
# # # # #         self.model.eval()
# # # # #         if self.config.training.use_text_proj:
# # # # #             self.text_proj.eval()
# # # # #         clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).cuda()
# # # # #         if self.config.training.use_text_proj:
# # # # #             clip_text_feat = self.text_proj(clip_text_feat)
# # # # #         per_cat_correct = torch.zeros(1156).cuda()
# # # # #         per_cat_count = torch.zeros(1156).cuda()
# # # # #         category2idx = self.objaverse_lvis_loader.dataset.category2idx
# # # # #         idx2category = {v: k for k, v in category2idx.items()}

# # # # #         logits_all = []
# # # # #         labels_all = []
# # # # #         with torch.no_grad():
# # # # #             for data in self.objaverse_lvis_loader:
# # # # #                 if not self.config.model.get("use_dense", False):
# # # # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # # # #                                            device=self.config.device, \
# # # # #                                            quantization_size=self.config.model.voxel_size)
# # # # #                 else:
# # # # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # # # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # # # #                 labels = data['category'].to(self.config.device)
# # # # #                 logits_all.append(logits.detach())
# # # # #                 labels_all.append(labels)

# # # # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "objaverse_dir"),
# # # # #                                                     logits_all, labels_all)

# # # # #         if self.rank == 0:
# # # # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # # # #             # calculate per class accuracy
# # # # #             for i in torch.unique(labels_all):
# # # # #                 idx = (labels_all == i)
# # # # #                 if idx.sum() > 0:
# # # # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # # # #                     per_cat_count[i] = idx.sum()

# # # # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # # # #             per_cat_acc = per_cat_correct / per_cat_count

# # # # #             if overall_acc > self.best_lvis_acc:
# # # # #                 self.best_lvis_acc = overall_acc
# # # # #                 # self.save_model('best_lvis')

# # # # #             logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# # # # #             logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # # # #                                                                                                 topk_acc[1].item(),
# # # # #                                                                                                 topk_acc[2].item()))
# # # # #             # wandb.log({"test_lvis/epoch": self.epoch,
# # # # #             #            "test_lvis/step": self.step,
# # # # #             #            "test_lvis/overall_acc": overall_acc,
# # # # #             #            "test_lvis/class_acc": per_cat_acc.mean(),
# # # # #             #            "test_lvis/top3_acc": topk_acc[1],
# # # # #             #            "test_lvis/top5_acc": topk_acc[2], })

# # # # #     def test_scanobjectnn(self):
# # # # #         self.model.eval()
# # # # #         if self.config.training.use_text_proj:
# # # # #             self.text_proj.eval()
# # # # #         clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
# # # # #         if self.config.training.use_text_proj:
# # # # #             clip_text_feat = self.text_proj(clip_text_feat)
# # # # #         per_cat_correct = torch.zeros(15).to(self.config.device)
# # # # #         per_cat_count = torch.zeros(15).to(self.config.device)
# # # # #         category2idx = self.scanobjectnn_loader.dataset.category2idx
# # # # #         idx2category = {v: k for k, v in category2idx.items()}

# # # # #         logits_all = []
# # # # #         labels_all = []
# # # # #         with torch.no_grad():
# # # # #             for data in self.scanobjectnn_loader:
# # # # #                 if not self.config.model.get("use_dense", False):
# # # # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # # # #                                            device=self.config.device, \
# # # # #                                            quantization_size=self.config.model.voxel_size)
# # # # #                 else:
# # # # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # # # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # # # #                 labels = data['category'].to(self.config.device)
# # # # #                 logits_all.append(logits.detach())
# # # # #                 labels_all.append(labels)

# # # # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scanobjectnn_dir"),
# # # # #                                                     logits_all, labels_all)

# # # # #         if self.rank == 0:

# # # # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # # # #             # calculate per class accuracy
# # # # #             for i in range(15):
# # # # #                 idx = (labels_all == i)
# # # # #                 if idx.sum() > 0:
# # # # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # # # #                     per_cat_count[i] = idx.sum()

# # # # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # # # #             per_cat_acc = per_cat_correct / per_cat_count

# # # # #             logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# # # # #             logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # # # #                                                                                                topk_acc[1].item(),
# # # # #                                                                                                topk_acc[2].item()))
# # # # #             # wandb.log({"test_scanobjectnn/epoch": self.epoch,
# # # # #             #            "test_scanobjectnn/step": self.step,
# # # # #             #            "test_scanobjectnn/overall_acc": overall_acc,
# # # # #             #            "test_scanobjectnn/class_acc": per_cat_acc.mean(),
# # # # #             #            "test_scanobjectnn/top3_acc": topk_acc[1],
# # # # #             #            "test_scanobjectnn/top5_acc": topk_acc[2], })


# # # # #     def test_scannet(self, scannet_loader):
# # # # #         self.model.eval()
# # # # #         if self.config.training.use_text_proj:
# # # # #             self.text_proj.eval()
# # # # #         clip_text_feat = torch.from_numpy(scannet_loader.dataset.clip_cat_feat).to(self.config.device)
# # # # #         if self.config.training.use_text_proj:
# # # # #             clip_text_feat = self.text_proj(clip_text_feat)
# # # # #         per_cat_correct = torch.zeros(19).to(self.config.device)
# # # # #         per_cat_count = torch.zeros(19).to(self.config.device)
# # # # #         category2idx = scannet_loader.dataset.category2idx
# # # # #         idx2category = {v: k for k, v in category2idx.items()}

# # # # #         logits_all = []
# # # # #         labels_all = []
# # # # #         with torch.no_grad():
# # # # #             for data in tqdm(scannet_loader):
# # # # #                 if not self.config.model.get("use_dense", False):
# # # # #                     pred_feat = self.model(data['xyz'], data['features'], \
# # # # #                                            device=self.config.device, \
# # # # #                                            quantization_size=self.config.model.voxel_size)
# # # # #                 else:
# # # # #                     pred_feat = self.model(data['xyz_dense'], data['features_dense'])
# # # # #                 logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
# # # # #                 labels = data['category'].to(self.config.device)
# # # # #                 logits_all.append(logits.detach())
# # # # #                 labels_all.append(labels)

# # # # #         logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, "scannet_dir"),
# # # # #                                                     logits_all, labels_all)
# # # # #         if self.rank == 0:

# # # # #             topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

# # # # #             # calculate per class accuracy
# # # # #             for i in range(19):
# # # # #                 idx = (logits_all == i)
# # # # #                 if idx.sum() > 0:
# # # # #                     per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
# # # # #                     per_cat_count[i] = idx.sum()

# # # # #             overall_acc = per_cat_correct.sum() / per_cat_count.sum()
# # # # #             per_cat_acc = per_cat_correct / per_cat_count


# # # # #             logging.info('Test Scannet: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
# # # # #             logging.info('Test Scannet: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
# # # # #                                                                                            topk_acc[1].item(),
# # # # #                                                                                                topk_acc[2].item()))