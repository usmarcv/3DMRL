import logging
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
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


class TAMM_Trainer(object):
    def __init__(self, rank, config, model, logit_scale, 
                 pretrained_image_adapter, pretrained_text_adapter, 
                 optimizer, scheduler, train_loader, 
                 modelnet40_loader, objaverse_lvis_loader, scanobjectnn_loader):    
                               
        self.rank = rank
        self.config = config
        
        # --- Modelo Treinável (Estágio 2) ---
        self.model = model # PointBERT (Cospe a dimensão final direta)
        
        self.pretrained_image_adapter = pretrained_image_adapter
        self.pretrained_text_adapter = pretrained_text_adapter
        self.logit_scale = logit_scale
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        
        self.epoch = 0
        self.step = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0
        self.config.ngpu = dist.get_world_size()

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if not self.config.training.get('use_openclip_optimizer_scheduler', False):
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        logging.info(f"Loaded Stage 2 checkpoint from {path} (Epoch: {self.epoch}, Step: {self.step})")

    def save_model(self, name):
        # Salva APENAS o modelo 3D, pois os professores do Estágio 1 já estão salvos
        torch.save({
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if not self.config.training.get('use_openclip_optimizer_scheduler', False) else None,
            "epoch": self.epoch,
            "step": self.step,
        }, os.path.join(self.config.ckpt_dir, f'{name}.pt'))

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



    # def train_one_epoch(self):  

    #     self.model.train()

    #     # Garante que os professores estejam em eval (segurança extra)
    #     if self.pretrained_image_adapter is not None:
    #         self.pretrained_image_adapter.eval()
    #     if self.pretrained_text_adapter is not None:
    #         self.pretrained_text_adapter.eval()

    #     mrl_dims = self.config.nesting_list

    #     epoch_img_loss_dim = {dim: [] for dim in mrl_dims}
    #     epoch_img_acc_dim = {dim: [] for dim in mrl_dims}
    #     epoch_txt_loss_dim = {dim: [] for dim in mrl_dims}
    #     epoch_txt_acc_dim = {dim: [] for dim in mrl_dims}

    #     text_contras_acc_list = []
    #     img_contras_acc_list = []

    #     if self.config.training.use_mask:
    #         k = self.config.dataset.negative_sample_num
    #         s = self.config.dataset.train_batch_size
    #         mask1 = np.eye(k * s).astype(np.bool_)
    #         mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool_)
    #         mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

    #     for data in tqdm(self.train_loader):
    #         self.step += 1
    #         self.optimizer.zero_grad()
    #         total_loss = 0.0
            
    #         # 1. Extração 3D Direta (PointBERT)
    #         if not self.config.model.get("use_dense", False):
    #             pred_feat_3d = self.model(data['xyz'], data['features'], device=self.config.device, quantization_size=self.config.model.voxel_size)
    #         else:
    #             pred_feat_3d = self.model(data['xyz_dense'], data['features_dense'])

    #         logit_scale = self.logit_scale(None)
            
    #         idx = data['has_text_idx']
    #         text_feat = torch.vstack(data['text_feat']).to(self.config.device)
    #         img_feat = torch.vstack(data['img_feat']).to(self.config.device)

    #         mask = None
    #         if self.config.training.use_mask:
    #             first_img = img_feat[:, :self.config.clip_embed_dim]
    #             img_text_sim = F.normalize(first_img, dim=-1) @ F.normalize(text_feat, dim=-1).T
    #             mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
    #             mask = torch.logical_or(mask, mask_other).detach()

    #         # ******************** ALINHAMENTO MRL IMAGEM ********************
    #         if self.config.dataset.num_imgs > 0 and self.pretrained_image_adapter is not None:
    #             img_acc_accum = 0.0
    #             for i in range(self.config.dataset.num_imgs):
    #                 single_image_feat = img_feat[:, i * self.config.clip_embed_dim : (i + 1) * self.config.clip_embed_dim]
                    
    #                 # Passa pelo professor de IMAGEM para organizar na matrioska
    #                 with torch.no_grad():
    #                     single_image_feat = self.pretrained_image_adapter(single_image_feat)

    #                 for dim in mrl_dims:
    #                     pred_slice = pred_feat_3d[:, :dim]
    #                     img_slice = single_image_feat[:, :dim]
                        
    #                     slice_loss, slice_acc = self.calc_contrastive_loss(pred_slice, img_slice, logit_scale, mask)
                        
    #                     total_loss += (slice_loss * self.config.training.lambda_img_contras) / len(mrl_dims)
    #                     img_acc_accum += slice_acc / len(mrl_dims)
                        
    #                     epoch_img_loss_dim[dim].append(slice_loss.item())
    #                     epoch_img_acc_dim[dim].append(slice_acc.item())

    #             img_acc_mean = img_acc_accum / self.config.dataset.num_imgs
    #             img_contras_acc_list.append(img_acc_mean.item())

    #         # ******************** ALINHAMENTO MRL TEXTO ********************
    #         if len(idx) > 0 and self.pretrained_text_adapter is not None:
    #             # Passa pelo professor de TEXTO para organizar na matrioska
    #             with torch.no_grad():
    #                 text_feat = self.pretrained_text_adapter(text_feat)

    #             txt_acc_accum = 0.0
    #             for dim in mrl_dims:
    #                 pred_slice = pred_feat_3d[idx, :dim]
    #                 txt_slice = text_feat[:, :dim]
                    
    #                 slice_loss, slice_acc = self.calc_contrastive_loss(pred_slice, txt_slice, logit_scale, mask)
                    
    #                 total_loss += (slice_loss * self.config.training.lambda_text_contras) / len(mrl_dims)
    #                 txt_acc_accum += slice_acc / len(mrl_dims)
                    
    #                 epoch_txt_loss_dim[dim].append(slice_loss.item())
    #                 epoch_txt_acc_dim[dim].append(slice_acc.item())

    #             text_contras_acc_list.append(txt_acc_accum.item())

    #         # Otimização (Atualiza APENAS o PointBERT)
    #         total_loss.backward()
    #         self.optimizer.step()

    #         if self.config.training.scheduler in ["cosine", "const"]:
    #             self.scheduler(self.step)
    #         else:
    #             self.scheduler.step()
        
    #     # --- LOGGING MRL ---
    #     if self.rank == 0:
    #         # logging.info(f'--- Resumo Estágio 2 | Época {self.epoch} ---')
    #         logging.info('Treino Média: Acc Txt: {0:.4f} | Acc Img: {1:.4f}'.format(
    #             np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
    #             np.mean(img_contras_acc_list) if len(img_contras_acc_list) > 0 else 0))

    #         header = f"{'Dim':<6} | {'Img Loss':<10} | {'Img Acc':<10} | {'Txt Loss':<10} | {'Txt Acc':<10}"
    #         logging.info("-" * len(header))
    #         logging.info(header)
    #         for dim in mrl_dims:
    #             i_loss = np.mean(epoch_img_loss_dim[dim]) if epoch_img_loss_dim[dim] else 0
    #             i_acc  = np.mean(epoch_img_acc_dim[dim]) if epoch_img_acc_dim[dim] else 0
    #             t_loss = np.mean(epoch_txt_loss_dim[dim]) if epoch_txt_loss_dim[dim] else 0
    #             t_acc  = np.mean(epoch_txt_acc_dim[dim]) if epoch_txt_acc_dim[dim] else 0
    #             logging.info(f"{dim:>4}d  | {i_loss:.4f}     | {i_acc:.4f}     | {t_loss:.4f}     | {t_acc:.4f}")
    #         logging.info("-" * len(header))


    def train_one_epoch(self):  

        self.model.train()

        # Garante que os professores estejam em eval
        if self.pretrained_image_adapter is not None:
            self.pretrained_image_adapter.eval()
        if self.pretrained_text_adapter is not None:
            self.pretrained_text_adapter.eval()

        mrl_dims = self.config.nesting_list

        epoch_img_loss_dim = {dim: [] for dim in mrl_dims}
        epoch_img_acc_dim = {dim: [] for dim in mrl_dims}
        epoch_txt_loss_dim = {dim: [] for dim in mrl_dims}
        epoch_txt_acc_dim = {dim: [] for dim in mrl_dims}

        text_contras_acc_list = []
        img_contras_acc_list = []

        for data in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            total_loss = 0.0
            
            # 1. Extração 3D Direta (PointBERT)
            if not self.config.model.get("use_dense", False):
                pred_feat_3d = self.model(data['xyz'], data['features'], device=self.config.device, quantization_size=self.config.model.voxel_size)
            else:
                pred_feat_3d = self.model(data['xyz_dense'], data['features_dense'])

            logit_scale = self.logit_scale(None)
            
            idx = data['has_text_idx']
            text_feat = torch.vstack(data['text_feat']).to(self.config.device)
            img_feat = torch.vstack(data['img_feat']).to(self.config.device)

            # --- CORREÇÃO: Máscaras Dinâmicas para Lotes Variáveis ---
            N = pred_feat_3d.shape[0] # Tamanho real do lote atual (ex: 32)
            M = len(idx)              # Tamanho filtrado para quem tem texto (ex: 30)

            img_mask = None
            txt_mask = None

            if self.config.training.use_mask:
                # 1. Cria a mask_other base baseada no KNN para o tamanho N
                k = self.config.dataset.negative_sample_num
                mask1 = np.eye(k * N).astype(np.bool_)
                mask2 = np.kron(np.eye(N), np.ones((k, k))).astype(np.bool_)
                mask_other_N = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)
                
                # A img_mask (NxN) usa diretamente a topologia 3D (KNN)
                img_mask = mask_other_N.detach()

                if M > 0:
                    # 2. Cria a txt_mask (MxM) filtrando a imagem e a mask_other
                    first_img_filtered = img_feat[idx, :self.config.clip_embed_dim]
                    img_text_sim = F.normalize(first_img_filtered, dim=-1) @ F.normalize(text_feat, dim=-1).T
                    
                    mask_sim_txt = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                    
                    # Corta a matriz mask_other_N apenas para as linhas/colunas do 'idx'
                    mask_other_M = mask_other_N[idx][:, idx] 
                    
                    txt_mask = torch.logical_or(mask_sim_txt, mask_other_M).detach()

            # ******************** ALINHAMENTO MRL IMAGEM ********************
            if self.config.dataset.num_imgs > 0 and self.pretrained_image_adapter is not None:
                img_acc_accum = 0.0
                for i in range(self.config.dataset.num_imgs):
                    single_image_feat = img_feat[:, i * self.config.clip_embed_dim : (i + 1) * self.config.clip_embed_dim]
                    
                    with torch.no_grad():
                        single_image_feat = self.pretrained_image_adapter(single_image_feat)

                    for dim in mrl_dims:
                        pred_slice = pred_feat_3d[:, :dim]
                        img_slice = single_image_feat[:, :dim]
                        
                        # Usa a img_mask (NxN)
                        slice_loss, slice_acc = self.calc_contrastive_loss(pred_slice, img_slice, logit_scale, img_mask)
                        
                        total_loss += (slice_loss * self.config.training.lambda_img_contras) / len(mrl_dims)
                        img_acc_accum += slice_acc / len(mrl_dims)
                        
                        epoch_img_loss_dim[dim].append(slice_loss.item())
                        epoch_img_acc_dim[dim].append(slice_acc.item())

                img_acc_mean = img_acc_accum / self.config.dataset.num_imgs
                img_contras_acc_list.append(img_acc_mean.item())

            # ******************** ALINHAMENTO MRL TEXTO ********************
            if M > 0 and self.pretrained_text_adapter is not None:
                with torch.no_grad():
                    text_feat = self.pretrained_text_adapter(text_feat)

                txt_acc_accum = 0.0
                for dim in mrl_dims:
                    pred_slice = pred_feat_3d[idx, :dim]
                    txt_slice = text_feat[:, :dim]
                    
                    # Usa a txt_mask (MxM)
                    slice_loss, slice_acc = self.calc_contrastive_loss(pred_slice, txt_slice, logit_scale, txt_mask)
                    
                    total_loss += (slice_loss * self.config.training.lambda_text_contras) / len(mrl_dims)
                    txt_acc_accum += slice_acc / len(mrl_dims)
                    
                    epoch_txt_loss_dim[dim].append(slice_loss.item())
                    epoch_txt_acc_dim[dim].append(slice_acc.item())

                text_contras_acc_list.append(txt_acc_accum.item())

            # Otimização
            total_loss.backward()
            self.optimizer.step()

            if self.config.training.scheduler in ["cosine", "const"]:
                self.scheduler(self.step)
            else:
                self.scheduler.step()
        
        # --- LOGGING MRL (Permanece igual) ---
        if self.rank == 0:
            logging.info('Treino Média: Acc Txt: {0:.4f} | Acc Img: {1:.4f}'.format(
                np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
                np.mean(img_contras_acc_list) if len(img_contras_acc_list) > 0 else 0))

            header = f"{'Dim':<6} | {'Img Loss':<10} | {'Img Acc':<10} | {'Txt Loss':<10} | {'Txt Acc':<10}"
            logging.info("-" * len(header))
            logging.info(header)
            for dim in mrl_dims:
                i_loss = np.mean(epoch_img_loss_dim[dim]) if epoch_img_loss_dim[dim] else 0
                i_acc  = np.mean(epoch_img_acc_dim[dim]) if epoch_img_acc_dim[dim] else 0
                t_loss = np.mean(epoch_txt_loss_dim[dim]) if epoch_txt_loss_dim[dim] else 0
                t_acc  = np.mean(epoch_txt_acc_dim[dim]) if epoch_txt_acc_dim[dim] else 0
                logging.info(f"{dim:>4}d  | {i_loss:.4f}     | {i_acc:.4f}     | {t_loss:.4f}     | {t_acc:.4f}")
            logging.info("-" * len(header))
            

    def accuracy(self, output, target, topk=(1,)):
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
                logging.info(f"Epoch: {self.epoch}")
            self.train_one_epoch()
            if epoch >= self.config.training.test_epoch:
                self.test_modelnet40()
                self.test_objaverse_lvis()
                self.test_scanobjectnn()
            if self.rank == 0:
                self.save_model('latest')
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model(f'epoch_{self.epoch}')

    def test_modelnet40(self):
        self.model.eval()
        # if self.config.training.use_text_proj:
        #     self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).to(self.config.device)
        # if self.config.training.use_text_proj:
        #     clip_text_feat = self.text_proj(clip_text_feat)

        category2idx = self.modelnet40_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_all = []
        labels_all = []
        with torch.no_grad():
            # if hasattr(self.mrl_nested_proj, 'module'):
            #     weight_matrix = self.mrl_nested_proj.module.weight
            # else:
            #     weight_matrix = self.mrl_nested_proj.weight
            for data in self.modelnet40_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                # pred_feat_projetado = F.linear(pred_feat, weight_matrix)
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(logits_all, labels_all)

        if self.rank == 0:

            dataset_size = len(self.modelnet40_loader.dataset)
            logits_all = logits_all[:dataset_size]
            labels_all = labels_all[:dataset_size]

            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))
            per_cat_correct = torch.zeros(40).to(self.config.device)
            per_cat_count = torch.zeros(40).to(self.config.device)
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
            #            "test/top5_acc": topk_acc[2], })                                topk_acc[2].item()))

            torch.save({
                        "logits": logits_all,
                        "labels": labels_all,
                        "overall_acc": overall_acc,
                        "class_acc": per_cat_acc.mean()
                         },
                        os.path.join(self.config.ckpt_dir,f"modelnet40_epoch_{self.epoch}.pth")
                    )


    def test_objaverse_lvis(self):
        self.model.eval()
        # if self.config.training.use_text_proj:
        #     self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).to(self.config.device)
        # if self.config.training.use_text_proj:
        #     clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(1156).to(self.config.device)
        per_cat_count = torch.zeros(1156).to(self.config.device)
        category2idx = self.objaverse_lvis_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_all = []
        labels_all = []
        with torch.no_grad():
            # if hasattr(self.mrl_nested_proj, 'module'):
            #     weight_matrix = self.mrl_nested_proj.module.weight
            # else:
            #     weight_matrix = self.mrl_nested_proj.weight
            for data in self.objaverse_lvis_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                # pred_feat_projetado = F.linear(pred_feat, weight_matrix)
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(logits_all, labels_all)

        if self.rank == 0:
            dataset_size = len(self.objaverse_lvis_loader.dataset)
            logits_all = logits_all[:dataset_size]
            labels_all = labels_all[:dataset_size]
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

            torch.save({
                        "logits": logits_all,
                        "labels": labels_all,
                        "overall_acc": overall_acc,
                        "class_acc": per_cat_acc.mean()
                         },
                        os.path.join(self.config.ckpt_dir,f"objaverse_lvis_epoch_{self.epoch}.pth")
                    )


    def test_scanobjectnn(self):
        self.model.eval()
        # if self.config.training.use_text_proj:
        #     self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        # if self.config.training.use_text_proj:
        #     clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count = torch.zeros(15).to(self.config.device)
        category2idx = self.scanobjectnn_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_all = []
        labels_all = []
        with torch.no_grad():
            # if hasattr(self.mrl_nested_proj, 'module'):
            #     weight_matrix = self.mrl_nested_proj.module.weight
            # else:
            #     weight_matrix = self.mrl_nested_proj.weight
            for data in self.scanobjectnn_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                # pred_feat_projetado = F.linear(pred_feat, weight_matrix)
                logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_all.append(logits.detach())
                labels_all.append(labels)

        logits_all, labels_all = merge_results_dist(logits_all, labels_all)

        if self.rank == 0:
            dataset_size = len(self.scanobjectnn_loader.dataset)
            logits_all = logits_all[:dataset_size]
            labels_all = labels_all[:dataset_size]
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
            torch.save({
                        "logits": logits_all,
                        "labels": labels_all,
                        "overall_acc": overall_acc,
                        "class_acc": per_cat_acc.mean()
                         },
                        os.path.join(self.config.ckpt_dir,f"scanobjectnn_epoch_{self.epoch}.pth")
                    )   


    def test_scannet(self, scannet_loader):
        self.model.eval()
        # if self.config.training.use_text_proj:
        #     self.text_proj.eval()
        clip_text_feat = torch.from_numpy(scannet_loader.dataset.clip_cat_feat).to(self.config.device)
        # if self.config.training.use_text_proj:
        #     clip_text_feat = self.text_proj(clip_text_feat)
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

        logits_all, labels_all = merge_results_dist(logits_all, labels_all)

        if self.rank == 0:
            dataset_size = len(self.scannet_loader.dataset)
            logits_all = logits_all[:dataset_size]
            labels_all = labels_all[:dataset_size]

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

            torch.save({
                        "logits": logits_all,  
                        "labels": labels_all,
                        "overall_acc": overall_acc,
                        "class_acc": per_cat_acc.mean()
                            },
                        os.path.join(self.config.ckpt_dir,f"scannet_epoch_{self.epoch}.pth"))