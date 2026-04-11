import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
from numpy import *
from tqdm import tqdm

from trainers.trainer_utils import merge_two_branch_results_dist
from trainers.trainer_utils import merge_results_dist


# from MRL import MRL_Linear_Layer, Matryoshka_CE_Loss, FixedFeatureLayer


# from trainers.mrl_trainer_adapters import MRLProjectionHeads

from torch import nn

import math
from collections import OrderedDict, defaultdict

from trainers.trainer_utils import merge_results_dist



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




class TrainerToMRL(object):
    def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, mrl_heads,
                 optimizer,
                 scheduler, train_loader, \
                 modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None):
        
        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.mrl_heads = mrl_heads
        # self.image_alignment_adapter = image_alignment_adapter
        # self.text_alignment_adapter = text_alignment_adapter
        # self.clip_adapter = clip_adapter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.epoch = 0
        self.step = 0
        self.alpha = 0.5
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0
	    
        if dist.is_initialized():
            self.config.ngpu = dist.get_world_size()
        else:
            self.config.ngpu = 1
        
        # self.mrl_dims = self.config.model.get("mrl_dims", None)
        # self.mrl = MRL_Linear_Layer(nesting_list=self.mrl_dims, num_classes=0).to(self.config.device)


    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])

        # self.image_alignment_adapter.load_state_dict(checkpoint['image_alignment_adapter'])
        # self.text_alignment_adapter.load_state_dict(checkpoint['text_alignment_adapter'])

        self.logit_scale.load_state_dict(checkpoint['logit_scale'])  # module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.scheduler == "default":
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1
        self.step = checkpoint['step']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

    def _get_module(self, module):
        return module.module if hasattr(module, "module") else module


    def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            # i=5
            # if i<4:
            feat1 = F.normalize(feat1, dim=1) #[B, D]
            print("feat1", feat1.shape, self.rank)
            feat2 = F.normalize(feat2, dim=1)
            print("feat2", feat2.shape, self.rank)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
            # print("logit", logits.shape, self.rank)
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy


    def mrl_loss(self, feat_pc, feat_clip, logit_scale=1, mask=None, lambdas=None):

        heads    = self._get_module(self.mrl_heads)
        mrl_dims = heads.mrl_dims

        if lambdas is None:
            lambdas = [1.0] * len(mrl_dims)

        projected = heads(feat_pc)  

        t = F.normalize(feat_clip, dim=-1) 

        total_loss   = torch.tensor(0.0, device=self.config.device)
        total_acc    = 0.0
        loss_per_dim = {}
        acc_per_dim  = {}

        for i, (s, dim) in enumerate(zip(projected, mrl_dims)):

            if self.config.ngpu > 1:
                all_s = torch.cat(torch.distributed.nn.all_gather(s), dim=0)  # (B*G, 1280)
                all_t = torch.cat(torch.distributed.nn.all_gather(t), dim=0)  # (B*G, 1280)
                logits = logit_scale * (all_s @ all_t.T)                      # (B*G, B*G)

                labels = torch.arange(logits.shape[0], device=self.config.device)
                loss_dim = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0

                B_local     = s.shape[0]
                local_start = self.rank * B_local
                local_end   = local_start + B_local
                local_logits = logits[local_start:local_end]
                local_labels = torch.arange(local_start, local_end, device=self.config.device)
                acc_dim = (local_logits.argmax(dim=1) == local_labels).float().mean()

            else:
                logits   = logit_scale * (s @ t.T)
                labels   = torch.arange(logits.shape[0], device=self.config.device)
                loss_dim = (
                    F.cross_entropy(logits, labels) +
                    F.cross_entropy(logits.T, labels)
                ) / 2.0
                acc_dim  = (logits.argmax(dim=1) == labels).float().mean()

            # ── acumula FORA do if/else, para ambos os casos ──────────────
            total_loss += lambdas[i] * loss_dim  
            total_acc  += acc_dim

            loss_per_dim[dim] = loss_dim.detach().item()
            acc_per_dim[dim]  = acc_dim.detach().item()

        total_loss = total_loss / sum(lambdas)

        return total_loss, total_acc / len(mrl_dims), loss_per_dim, acc_per_dim


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


    def train_one_epoch(self):
        self.model.train()
        if self.config.training.use_text_proj: #False
            self.text_proj.train()
        if self.config.training.use_image_proj: #False
            self.image_proj.train()

        text_contras_acc_list = []
        img_contras_acc_list = []
    
        #mrl metrics per dimension
        epoch_img_loss_dim = defaultdict(list)
        epoch_img_acc_dim = defaultdict(list)
        epoch_txt_loss_dim = defaultdict(list)
        epoch_txt_acc_dim = defaultdict(list)

        lambdas = getattr(self.config.mrl, "lambdas", None)

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
                pred_feat = self.model(data['xyz'], data['features'], device=self.config.device,
                                       quantization_size=self.config.model.voxel_size)
            else:
                pred_feat = self.model(data['xyz_dense'].to(self.config.device), \
                                       data['features_dense'].to(self.config.device))
            
            logit_scale = self.logit_scale(None)
            text_feat = torch.vstack(data['text_feat']) # image feature from dataset
            img_feat = torch.vstack(data['img_feat']) # text feature

            if self.config.training.use_mask:
                img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
                mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                mask = torch.logical_or(mask, mask_other).detach()
            else:
                mask = None

            text_feat = torch.vstack(data['text_feat']).to(self.config.device)
            img_feat = torch.vstack(data['img_feat']).to(self.config.device)
            idx = data['has_text_idx']
            
            if self.config.dataset.num_imgs > 0:       
                for i in range(self.config.dataset.num_imgs):
                    single_img_feat = img_feat[:, i * self.config.clip_embed_dim: (i + 1) * self.config.clip_embed_dim].to(self.config.device)

                    img_contras_loss, img_contras_acc, \
                    img_loss_dims, img_acc_dims = \
                    self.mrl_loss(pred_feat, single_img_feat,
                                 logit_scale=logit_scale,
                               mask=mask, lambdas=lambdas)
                        
                        
                    loss += img_contras_loss * self.config.training.lambda_img_contras 
                    img_contras_acc_list.append(img_contras_acc.item())

                #calcula as métricas por dimensão
                for d, val in img_loss_dims.items():
                    epoch_img_loss_dim[d].append(val)
                   
                for d, val in img_acc_dims.items():
                    epoch_img_acc_dim[d].append(val)


                for i in range(self.config.dataset.num_texts):

                    single_text_feat = text_feat[:, i * self.config.clip_embed_dim: (i + 1) * self.config.clip_embed_dim]
                    single_text_feat = single_text_feat.to(self.config.device)

                    text_contras_loss, text_contras_acc, txt_loss_dims, txt_acc_dims = self.mrl_loss(pred_feat[idx], single_text_feat,
                                                logit_scale=logit_scale, mask=mask, lambdas=lambdas)
                        
                    loss += text_contras_loss * self.config.training.lambda_text_contras
                    text_contras_acc_list.append(text_contras_acc.item())
                    # text_contras_acc_list.append(float(text_contras_acc))

                for d, val in txt_loss_dims.items():
                    epoch_txt_loss_dim[d].append(val)
                for d, val in txt_acc_dims.items():
                    epoch_txt_acc_dim[d].append(val)


            loss.backward()
            self.optimizer.step()

            if self.config.training.scheduler == "cosine" or self.config.training.scheduler == "const":
                self.scheduler(self.step)
            else:
                self.scheduler.step()

       
        if self.rank == 0:
            logging.info('Train avg: image_contrast_acc: {0} text_contrast_acc: {1}' \
                         .format(np.mean(img_contras_acc_list)  \
                                 np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0))
            
            # --- Tabela MRL Formatada ---
            header = f"{'Dim':<6} | {'Img Loss':<10} | {'Img Acc':<10} | {'Txt Loss':<10} | {'Txt Acc':<10}"
            logging.info("-" * len(header))
            logging.info(header)
            logging.info("-" * len(header))
            
            # Garante que iteramos sobre todas as dimensões presentes
            all_dims = sorted(set(epoch_img_loss_dim.keys()) | set(epoch_txt_loss_dim.keys()))
            
            for d in all_dims:
                # Calcula médias com segurança (get retorna lista vazia se chave não existir)
                i_loss = np.mean(epoch_img_loss_dim.get(d, [0]))
                i_acc  = np.mean(epoch_img_acc_dim.get(d, [0]))
                t_loss = np.mean(epoch_txt_loss_dim.get(d, [0]))
                t_acc  = np.mean(epoch_txt_acc_dim.get(d, [0]))
                
                logging.info(f"{d:<6} | {i_loss:.4f}     | {i_acc:.4f}     | {t_loss:.4f}     | {t_acc:.4f}")
            
            logging.info("-" * len(header))


    def save_model(self, name):
        torch.save({
            "state_dict": self.model.state_dict(),
            "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.scheduler == "default" else None,
            "epoch": self.epoch,
            "step": self.step,
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


    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()

            if epoch > self.config.training.test_epoch:
                self.test_objaverse_lvis()
                self.test_modelnet40()
                self.test_scanobjectnn()
            # if self.rank == 0:
            # self.save_model('latest')
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))

                
    def test_modelnet40(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()

        clip_text_feat = torch.from_numpy(
            self.modelnet40_loader.dataset.clip_cat_feat
        ).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

        logits_all = []
        labels_all = []

        with torch.no_grad():
            for data in tqdm(self.modelnet40_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(
                        data['xyz'], data['features'],
                        device=self.config.device,
                        quantization_size=self.config.model.voxel_size
                    )
                else:
                    pred_feat = self.model(
                        data['xyz_dense'].to(self.config.device),
                        data['features_dense'].to(self.config.device)
                    )

                # inferência com a head completa (maior dim)
                shape_emb = self._get_shape_embedding(pred_feat)  # (B, 1280)
                logits    = shape_emb @ F.normalize(clip_text_feat, dim=-1).T
                labels    = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(logits_all, labels_all)

        if self.rank == 0:
            if logits_all is None:
                return

            dataset_size = len(self.modelnet40_loader.dataset)
            logits_all   = logits_all[:dataset_size]
            labels_all   = labels_all[:dataset_size]

            topk_acc, _     = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))
            per_cat_correct = torch.zeros(40).to(self.config.device)
            per_cat_count   = torch.zeros(40).to(self.config.device)

            for i in range(40):
                idx = labels_all == i
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i]   = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            if overall_acc > self.best_modelnet40_overall_acc:
                self.best_modelnet40_overall_acc = overall_acc
            if per_cat_acc.mean() > self.best_modelnet40_class_acc:
                self.best_modelnet40_class_acc = per_cat_acc.mean()

            logging.info(
                'Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(
                    overall_acc, self.best_modelnet40_overall_acc,
                    per_cat_acc.mean(), self.best_modelnet40_class_acc,
                )
            )
            logging.info(
                'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(
                    topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
                )
            )
            torch.save(
                {"logits": logits_all, "labels": labels_all,
                "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
                os.path.join(self.config.ckpt_dir, f"modelnet40_epoch_{self.epoch}.pth"),
            )

    def test_objaverse_lvis(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()

        clip_text_feat = torch.from_numpy(
            self.objaverse_lvis_loader.dataset.clip_cat_feat
        ).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

        per_cat_correct = torch.zeros(1156).to(self.config.device)
        per_cat_count   = torch.zeros(1156).to(self.config.device)

        logits_all = []
        labels_all = []

        with torch.no_grad():
            for data in tqdm(self.objaverse_lvis_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(
                        data['xyz'], data['features'],
                        device=self.config.device,
                        quantization_size=self.config.model.voxel_size
                    )
                else:
                    pred_feat = self.model(
                        data['xyz_dense'].to(self.config.device),
                        data['features_dense'].to(self.config.device)
                    )

                shape_emb = self._get_shape_embedding(pred_feat)  # (B, 1280)
                logits    = shape_emb @ F.normalize(clip_text_feat, dim=-1).T
                labels    = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(logits_all, labels_all)

        if self.rank == 0:
            if logits_all is None:
                return

            dataset_size = len(self.objaverse_lvis_loader.dataset)
            logits_all   = logits_all[:dataset_size]
            labels_all   = labels_all[:dataset_size]

            topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))

            for i in torch.unique(labels_all):
                idx = labels_all == i
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i]   = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            if overall_acc > self.best_lvis_acc:
                self.best_lvis_acc = overall_acc
                self.save_model('best_lvis')

            logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(
                overall_acc, per_cat_acc.mean()))
            logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(
                topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))

            torch.save(
                {"logits": logits_all, "labels": labels_all,
                "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
                os.path.join(self.config.ckpt_dir, f"objaverse_lvis_epoch_{self.epoch}.pth"),
            )


    def test_scanobjectnn(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()

        clip_text_feat = torch.from_numpy(
            self.scanobjectnn_loader.dataset.clip_cat_feat
        ).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count   = torch.zeros(15).to(self.config.device)

        logits_all = []
        labels_all = []

        with torch.no_grad():
            for data in self.scanobjectnn_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(
                        data['xyz'], data['features'],
                        device=self.config.device,
                        quantization_size=self.config.model.voxel_size
                    )
                else:
                    pred_feat = self.model(
                        data['xyz_dense'].to(self.config.device),
                        data['features_dense'].to(self.config.device)
                    )

                shape_emb = self._get_shape_embedding(pred_feat)  # (B, 1280)
                logits    = shape_emb @ F.normalize(clip_text_feat, dim=-1).T
                labels    = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(logits_all, labels_all)

        if self.rank == 0:
            if logits_all is None:
                return

            dataset_size = len(self.scanobjectnn_loader.dataset)
            logits_all   = logits_all[:dataset_size]
            labels_all   = labels_all[:dataset_size]

            topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))

            for i in range(15):
                idx = labels_all == i
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i]   = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(
                overall_acc, per_cat_acc.mean()))
            logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(
                topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))

            torch.save(
                {"logits": logits_all, "labels": labels_all,
                "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
                os.path.join(self.config.ckpt_dir, f"scanobjectnn_epoch_{self.epoch}.pth"),
            )   

    # def test_objaverse_lvis(self):
    #     self.model.eval()
    #     # self.text_alignment_adapter.eval()
    #     # self.image_alignment_adapter.eval()
    #     if self.config.training.use_text_proj:
    #         self.text_proj.eval()
    #     clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).cuda()
    #     if self.config.training.use_text_proj:
    #         clip_text_feat = self.text_proj(clip_text_feat)
    #     per_cat_correct = torch.zeros(1156).cuda()
    #     per_cat_count = torch.zeros(1156).cuda()
    #     category2idx = self.objaverse_lvis_loader.dataset.category2idx
    #     idx2category = {v: k for k, v in category2idx.items()}

    #     logits_image_all = []
    #     logits_text_all = []
    #     labels_all = []
    #     with torch.no_grad():
    #         for data in tqdm(self.objaverse_lvis_loader):
    #             if not self.config.model.get("use_dense", False):
    #                 pred_feat = self.model(data['xyz'], data['features'], \
    #                                        device=self.config.device, \
    #                                        quantization_size=self.config.model.voxel_size)
    #             else:
    #                 pred_feat = self.model(data['xyz_dense'].to(self.config.device), data['features_dense'].to(self.config.device))

    #             # pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
    #             # pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)
    #             # print("pred_feat_text", pred_feat_text.shape)
    #             # print("clip_text_feat", clip_text_feat.shape)
    #             logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
    #             logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T
    #             labels = data['category'].to(self.config.device)
    #             logits_image_all.append(logits_image.detach())
    #             logits_text_all.append(logits_text.detach())
    #             labels_all.append(labels)

    #     logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
    #         os.path.join(self.config.ckpt_dir, "objaverse_dir"),
    #         logits_image_all, logits_text_all, labels_all)

    #     if self.rank == 0:

    #         logits_all = 1 * logits_text_all + self.alpha * logits_image_all
    #         topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

    #         # calculate per class accuracy
    #         for i in torch.unique(labels_all):
    #             idx = (labels_all == i)
    #             if idx.sum() > 0:
    #                 per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
    #                 per_cat_count[i] = idx.sum()

    #         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    #         per_cat_acc = per_cat_correct / per_cat_count

    #         if overall_acc > self.best_lvis_acc:
    #             self.best_lvis_acc = overall_acc
    #             self.save_model('best_lvis')

    #         logging.info(
    #             'Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
    #         logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
    #                                                                                             topk_acc[1].item(),
    #                                                                                             topk_acc[2].item()))

    # def test_scanobjectnn(self):
    #     self.model.eval()
    #     # self.text_alignment_adapter.eval()
    #     # self.image_alignment_adapter.eval()
    #     if self.config.training.use_text_proj:
    #         self.text_proj.eval()
    #     clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
    #     if self.config.training.use_text_proj:
    #         clip_text_feat = self.text_proj(clip_text_feat)
    #     per_cat_correct = torch.zeros(15).to(self.config.device)
    #     per_cat_count = torch.zeros(15).to(self.config.device)
    #     category2idx = self.scanobjectnn_loader.dataset.category2idx
    #     idx2category = {v: k for k, v in category2idx.items()}

    #     logits_image_all = []
    #     logits_text_all = []
    #     labels_all = []
    #     with torch.no_grad():
    #         for data in self.scanobjectnn_loader:
    #             if not self.config.model.get("use_dense", False):
    #                 pred_feat = self.model(data['xyz'], data['features'], \
    #                                        device=self.config.device, \
    #                                        quantization_size=self.config.model.voxel_size)
    #             else:
    #                 pred_feat = self.model(data['xyz_dense'].to(self.config.device), data['features_dense'].to(self.config.device))

    #             # pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
    #             # pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)

    #             logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
    #             logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T

    #             labels = data['category'].to(self.config.device)
    #             logits_image_all.append(logits_image.detach())
    #             logits_text_all.append(logits_text.detach())
    #             labels_all.append(labels)

    #     logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
    #         os.path.join(self.config.ckpt_dir, "scanobjectnn_dir"),
    #         logits_image_all, logits_text_all, labels_all)

    #     if self.rank == 0:

    #         logits_all = 1 * logits_text_all + self.alpha * logits_image_all

    #         topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

    #         # calculate per class accuracy
    #         for i in range(15):
    #             idx = (labels_all == i)
    #             if idx.sum() > 0:
    #                 per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
    #                 per_cat_count[i] = idx.sum()

    #         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    #         per_cat_acc = per_cat_correct / per_cat_count

    #         logging.info(
    #             'Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
    #         logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
    #                                                                                            topk_acc[1].item(),
    #                                                                                            topk_acc[2].item()))

