import logging
import os
import numpy as np
import torch
import math
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
from numpy import *
from tqdm import tqdm
from torch import nn
from torch.amp import autocast
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

        self.precision = self.config.training.precision
        if self.precision == "bf16":
            self.dtype = torch.bfloat16
        elif self.precision == "fp16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        use_scaler = (self.precision == "fp16")
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
        

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        
        self._get_module(self.model).load_state_dict(checkpoint['state_dict'])
        self._get_module(self.logit_scale).load_state_dict(checkpoint['logit_scale'])
        
        # Carrega os cabeçalhos MRL
        if 'mrl_heads' in checkpoint:
            self._get_module(self.mrl_heads).load_state_dict(checkpoint['mrl_heads'])
            
        # Carrega as projeções (se existirem no checkpoint e no config)
        if self.config.training.use_text_proj and 'text_proj' in checkpoint:
            self._get_module(self.text_proj).load_state_dict(checkpoint['text_proj'])
        if self.config.training.use_image_proj and 'image_proj' in checkpoint:
            self._get_module(self.image_proj).load_state_dict(checkpoint['image_proj'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.scheduler == "default":
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
        self.epoch = checkpoint['epoch'] + 1
        self.step = checkpoint['step']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))


    def _get_module(self, module):
        return module.module if hasattr(module, "module") else module
        

    def mrl_loss(self, feat_pc, feat_clip, logit_scale=1, mask=None, lambdas=None):

        # Obtém a nova MRL_Projection_Layer
        heads = self._get_module(self.mrl_heads)
        nesting_list = heads.nesting_list # Agora usamos o nome padrão
        
        if lambdas is None:
            lambdas = [1.0] * len(nesting_list)

        # O forward da nova classe já devolve a tupla de embeddings projetados!
        projected = heads(feat_pc)  

        t = F.normalize(feat_clip, dim=-1) 

        total_loss   = torch.tensor(0.0, device=self.config.device)
        total_acc    = 0.0
        loss_per_dim = {}
        acc_per_dim  = {}

        for i, (s, dim) in enumerate(zip(projected, nesting_list)):
            
            # Normalizamos o embedding projetado (s) antes da similaridade
            s = F.normalize(s, dim=-1)

            if self.config.ngpu > 1:
                all_s = torch.cat(torch.distributed.nn.all_gather(s), dim=0)  
                all_t = torch.cat(torch.distributed.nn.all_gather(t), dim=0)  
                logits = logit_scale * (all_s @ all_t.T)                      

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

            total_loss += lambdas[i] * loss_dim  
            total_acc  += acc_dim

            loss_per_dim[dim] = loss_dim.detach().item()
            acc_per_dim[dim]  = acc_dim.detach().item()

        total_loss = total_loss / sum(lambdas)

        return total_loss, total_acc / len(nesting_list), loss_per_dim, acc_per_dim


    def _get_shape_embedding(self, feat_pc, dim=None):
        """
        Projeção para inferência zero-shot com suporte ao modo eficiente.
        """
        heads = self._get_module(self.mrl_heads)
        nesting_list = heads.nesting_list

        # Se dim não for passado, assume a dimensão máxima (último elemento)
        if dim is None:
            dim = nesting_list[-1]

        # Fatiamos a entrada da Point Cloud na dimensão desejada
        x_slice = feat_pc[:, :dim]

        if heads.efficient:
            # Modo eficiente: Fatiar manualmente os pesos da única camada grande
            weight_slice = heads.proj_0.weight[:, :dim]
            z = torch.matmul(x_slice, weight_slice.t())
            
            if heads.proj_0.bias is not None:
                z += heads.proj_0.bias
        else:
            # Modo padrão: Usar a camada específica dessa dimensão
            idx = nesting_list.index(dim)
            head = getattr(heads, f"proj_{idx}")
            z = head(x_slice)

        # Retorna o embedding normalizado para a busca por cosseno
        return F.normalize(z, dim=-1)


    def train_one_epoch(self):
        self.model.train()
        if self.config.training.use_text_proj: #False
            self.text_proj.train()
        if self.config.training.use_image_proj: #False
            self.image_proj.train()

        text_contras_acc_list = []
        img_contras_acc_list = []
    
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

            with torch.autocast(device_type=self.config.device, dtype=self.dtype, enabled=(self.dtype != torch.float32)):

                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], device=self.config.device,
                                        quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                
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


            # loss.backward()
            self.scaler.scale(loss).backward()
            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.config.training.scheduler == "cosine" or self.config.training.scheduler == "const":
                self.scheduler(self.step)
            else:
                self.scheduler.step()

       
        if self.rank == 0:
            logging.info('Treino Média: Acc Txt: {0:.4f} | Acc Img: {1:.4f}'.format(
                np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
                np.mean(img_contras_acc_list) if len(img_contras_acc_list) > 0 else 0))
            
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
        checkpoint = {
            "state_dict": self._get_module(self.model).state_dict(),
            "logit_scale": self._get_module(self.logit_scale).state_dict(),
            "mrl_heads": self._get_module(self.mrl_heads).state_dict(), # <-- ADICIONADO
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.scheduler == "default" else None,
            "epoch": self.epoch,
            "step": self.step,
        }
        
        if self.config.training.use_text_proj:
            checkpoint["text_proj"] = self._get_module(self.text_proj).state_dict()
        if self.config.training.use_image_proj:
            checkpoint["image_proj"] = self._get_module(self.image_proj).state_dict()

        torch.save(checkpoint, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))


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
                logging.info("Precision used for training: {}".format(self.precision))
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()

            if epoch > self.config.training.test_epoch:
                self.test_modelnet40()
                self.test_objaverse_lvis()
                self.test_scanobjectnn()
            # if self.rank == 0:
            # self.save_model('latest')
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))


    def test_modelnet40(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        if self.config.training.use_image_proj:
            self.image_proj.eval()
        self._get_module(self.mrl_heads).eval()

        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

        heads = self._get_module(self.mrl_heads)
        nesting_list = heads.nesting_list

        logits_all = {dim: [] for dim in nesting_list}
        labels_all = []

        with torch.no_grad():
            for data in tqdm(self.modelnet40_loader):
                
                # =====================================================================
                # AUTOCAST APENAS NO FORWARD DO MODELO
                # =====================================================================
                with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.dtype != torch.float32)):
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

                    # Extrai o embedding completo UMA VEZ (tamanho máximo)
                    shape_emb_full = self._get_shape_embedding(pred_feat)  
                # =====================================================================

                # Converte para FP32 para evitar erro na multiplicação com o texto
                shape_emb_full = shape_emb_full.float()

                labels = data['category'].to(self.config.device)
                labels_all.append(labels)

                for dim in nesting_list:
                    # Fatia a Point Cloud e o Texto para a mesma dimensão do MRL
                    shape_emb_dim = shape_emb_full[:, :dim]  
                    clip_text_dim = clip_text_feat[:, :dim].float() # Garante que o texto está em FP32
                    
                    logits = shape_emb_dim @ F.normalize(clip_text_dim, dim=-1).T
                    logits_all[dim].append(logits.detach())

        merged_logits = {}
        final_labels = None
        for dim in nesting_list:
            merged_l, merged_labels = merge_results_dist(logits_all[dim], labels_all)
            merged_logits[dim] = merged_l
            if final_labels is None:
                final_labels = merged_labels 
        labels_all = final_labels 

        if self.rank == 0:
            if labels_all is None:
                return

            dataset_size = len(self.modelnet40_loader.dataset)
            labels_all = labels_all[:dataset_size]
            results_to_save = {"labels": labels_all, "dims": {}}

            logging.info("="*60)
            logging.info("Test ModelNet40 per Dimension (Zero-Shot)")
            header = f"{'Dim':<6} | {'Overall Acc':<11} | {'Class Acc':<10} | {'Top-1':<7} | {'Top-3':<7} | {'Top-5':<7}"
            logging.info("-" * len(header))
            logging.info(header)
            logging.info("-" * len(header))

            for dim in nesting_list:
                logits_dim = merged_logits[dim][:dataset_size]
                topk_acc, _ = self.accuracy(logits_dim, labels_all, topk=(1, 3, 5))
                
                per_cat_correct = torch.zeros(40).to(self.config.device)
                per_cat_count   = torch.zeros(40).to(self.config.device)

                for i in range(40):
                    idx = labels_all == i
                    if idx.sum() > 0:
                        per_cat_correct[i] = (logits_dim[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                        per_cat_count[i]   = idx.sum()

                valid_cats = per_cat_count > 0 
                overall_acc = (per_cat_correct.sum() / per_cat_count.sum()).item()
                per_cat_acc = (per_cat_correct[valid_cats] / per_cat_count[valid_cats]).mean().item()

                # Atualiza os melhores resultados apenas usando a dimensão máxima do MRL
                if dim == nesting_list[-1]: 
                    if overall_acc > self.best_modelnet40_overall_acc:
                        self.best_modelnet40_overall_acc = overall_acc
                    if per_cat_acc > self.best_modelnet40_class_acc:
                        self.best_modelnet40_class_acc = per_cat_acc

                logging.info(f"{dim:<6} | {overall_acc:<11.4f} | {per_cat_acc:<10.4f} | {topk_acc[0].item():<7.2f} | {topk_acc[1].item():<7.2f} | {topk_acc[2].item():<7.2f}")

                results_to_save["dims"][dim] = {
                    "logits": logits_dim, "overall_acc": overall_acc, "class_acc": per_cat_acc
                }

            logging.info("-" * len(header))
            logging.info(f"Best (Max Dim) Overall: {self.best_modelnet40_overall_acc:.4f} | Class: {self.best_modelnet40_class_acc:.4f}")
            logging.info("="*60)
            torch.save(results_to_save, os.path.join(self.config.ckpt_dir, f"modelnet40_epoch_{self.epoch}.pth"))


    def test_objaverse_lvis(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        if self.config.training.use_image_proj:
            self.image_proj.eval()
        self._get_module(self.mrl_heads).eval()

        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

        heads = self._get_module(self.mrl_heads)
        nesting_list = heads.nesting_list

        logits_all = {dim: [] for dim in nesting_list}
        labels_all = []

        with torch.no_grad():
            for data in tqdm(self.objaverse_lvis_loader):
                
                # =====================================================================
                # AUTOCAST NO FORWARD
                # =====================================================================
                with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.dtype != torch.float32)):
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

                    shape_emb_full = self._get_shape_embedding(pred_feat)  
                # =====================================================================

                # Converte para FP32
                shape_emb_full = shape_emb_full.float()

                labels = data['category'].to(self.config.device)
                labels_all.append(labels)

                for dim in nesting_list:
                    shape_emb_dim = shape_emb_full[:, :dim]  
                    clip_text_dim = clip_text_feat[:, :dim].float() # Garante FP32
                    
                    logits = shape_emb_dim @ F.normalize(clip_text_dim, dim=-1).T
                    logits_all[dim].append(logits.detach())

        merged_logits = {}
        final_labels = None
        for dim in nesting_list:
            merged_l, merged_labels = merge_results_dist(logits_all[dim], labels_all)
            merged_logits[dim] = merged_l
            if final_labels is None:
                final_labels = merged_labels 
        labels_all = final_labels 

        if self.rank == 0:
            if labels_all is None:
                return

            dataset_size = len(self.objaverse_lvis_loader.dataset)
            labels_all = labels_all[:dataset_size]
            results_to_save = {"labels": labels_all, "dims": {}}

            logging.info("="*60)
            logging.info("Test ObjaverseLVIS per Dimension (Zero-Shot)")
            header = f"{'Dim':<6} | {'Overall Acc':<11} | {'Class Acc':<10} | {'Top-1':<7} | {'Top-3':<7} | {'Top-5':<7}"
            logging.info("-" * len(header))
            logging.info(header)
            logging.info("-" * len(header))

            if not hasattr(self, 'best_lvis_class_acc'):
                self.best_lvis_class_acc = 0.0

            for dim in nesting_list:
                logits_dim = merged_logits[dim][:dataset_size]
                topk_acc, _ = self.accuracy(logits_dim, labels_all, topk=(1, 3, 5))
                
                per_cat_correct = torch.zeros(1156).to(self.config.device)
                per_cat_count   = torch.zeros(1156).to(self.config.device)

                for i in torch.unique(labels_all):
                    idx = labels_all == i
                    if idx.sum() > 0:
                        per_cat_correct[i] = (logits_dim[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                        per_cat_count[i]   = idx.sum()

                valid_cats = per_cat_count > 0 
                overall_acc = (per_cat_correct.sum() / per_cat_count.sum()).item()
                per_cat_acc = (per_cat_correct[valid_cats] / per_cat_count[valid_cats]).mean().item()

                if dim == nesting_list[-1]: 
                    is_best = False
                    if overall_acc > self.best_lvis_acc:
                        self.best_lvis_acc = overall_acc
                        is_best = True
                    if per_cat_acc > self.best_lvis_class_acc:
                        self.best_lvis_class_acc = per_cat_acc
                        is_best = True
                        
                    if is_best:
                        self.save_model('best_lvis')

                logging.info(f"{dim:<6} | {overall_acc:<11.4f} | {per_cat_acc:<10.4f} | {topk_acc[0].item():<7.2f} | {topk_acc[1].item():<7.2f} | {topk_acc[2].item():<7.2f}")

                results_to_save["dims"][dim] = {
                    "logits": logits_dim, "overall_acc": overall_acc, "class_acc": per_cat_acc
                }

            logging.info("-" * len(header))
            logging.info(f"Best (Max Dim) Overall: {self.best_lvis_acc:.4f} | Class: {self.best_lvis_class_acc:.4f}")
            logging.info("="*60)
            torch.save(results_to_save, os.path.join(self.config.ckpt_dir, f"objaverse_lvis_epoch_{self.epoch}.pth"))


    def test_scanobjectnn(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        if self.config.training.use_image_proj:
            self.image_proj.eval()
        self._get_module(self.mrl_heads).eval()

        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

        heads = self._get_module(self.mrl_heads)
        nesting_list = heads.nesting_list

        logits_all = {dim: [] for dim in nesting_list}
        labels_all = []

        with torch.no_grad():
            for data in self.scanobjectnn_loader:
                
                # =====================================================================
                # AUTOCAST NO FORWARD
                # =====================================================================
                with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=(self.dtype != torch.float32)):
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

                    shape_emb_full = self._get_shape_embedding(pred_feat)  
                # =====================================================================

                # Converte para FP32
                shape_emb_full = shape_emb_full.float()

                labels = data['category'].to(self.config.device)
                labels_all.append(labels)

                for dim in nesting_list:
                    shape_emb_dim = shape_emb_full[:, :dim]  
                    clip_text_dim = clip_text_feat[:, :dim].float() # Garante FP32
                    
                    logits = shape_emb_dim @ F.normalize(clip_text_dim, dim=-1).T
                    logits_all[dim].append(logits.detach())

        merged_logits = {}
        final_labels = None
        for dim in nesting_list:
            merged_l, merged_labels = merge_results_dist(logits_all[dim], labels_all)
            merged_logits[dim] = merged_l
            if final_labels is None:
                final_labels = merged_labels 
        labels_all = final_labels 

        if self.rank == 0:
            if labels_all is None:
                return

            dataset_size = len(self.scanobjectnn_loader.dataset)
            labels_all = labels_all[:dataset_size]
            results_to_save = {"labels": labels_all, "dims": {}}

            logging.info("="*60)
            logging.info("Test ScanObjectNN per Dimension (Zero-Shot)")
            header = f"{'Dim':<6} | {'Overall Acc':<11} | {'Class Acc':<10} | {'Top-1':<7} | {'Top-3':<7} | {'Top-5':<7}"
            logging.info("-" * len(header))
            logging.info(header)
            logging.info("-" * len(header))

            for dim in nesting_list:
                logits_dim = merged_logits[dim][:dataset_size]
                topk_acc, _ = self.accuracy(logits_dim, labels_all, topk=(1, 3, 5))
                
                per_cat_correct = torch.zeros(15).to(self.config.device)
                per_cat_count   = torch.zeros(15).to(self.config.device)

                for i in range(15):
                    idx = labels_all == i
                    if idx.sum() > 0:
                        per_cat_correct[i] = (logits_dim[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                        per_cat_count[i]   = idx.sum()

                valid_cats = per_cat_count > 0 
                overall_acc = (per_cat_correct.sum() / per_cat_count.sum()).item()
                per_cat_acc = (per_cat_correct[valid_cats] / per_cat_count[valid_cats]).mean().item()

                logging.info(f"{dim:<6} | {overall_acc:<11.4f} | {per_cat_acc:<10.4f} | {topk_acc[0].item():<7.2f} | {topk_acc[1].item():<7.2f} | {topk_acc[2].item():<7.2f}")

                results_to_save["dims"][dim] = {
                    "logits": logits_dim, "overall_acc": overall_acc, "class_acc": per_cat_acc
                }

            logging.info("-" * len(header))
            logging.info("="*60)
            torch.save(results_to_save, os.path.join(self.config.ckpt_dir, f"scanobjectnn_epoch_{self.epoch}.pth"))

    # def test_modelnet40(self):
    #     self.model.eval()
    #     if self.config.training.use_text_proj:
    #         self.text_proj.eval()
    #     clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).to(self.config.device)
    #     if self.config.training.use_text_proj:
    #         clip_text_feat = self.text_proj(clip_text_feat)

    #     logits_all = []
    #     labels_all = []

    #     with torch.no_grad():
    #         for data in tqdm(self.modelnet40_loader):
    #             if not self.config.model.get("use_dense", False):
    #                 pred_feat = self.model(
    #                     data['xyz'], data['features'],
    #                     device=self.config.device,
    #                     quantization_size=self.config.model.voxel_size
    #                 )
    #             else:
    #                 pred_feat = self.model(
    #                     data['xyz_dense'].to(self.config.device),
    #                     data['features_dense'].to(self.config.device)
    #                 )

    #             # inferência com a head completa (maior dim)
    #             shape_emb = self._get_shape_embedding(pred_feat)  # (B, 1280)
    #             logits    = shape_emb @ F.normalize(clip_text_feat, dim=-1).T
    #             labels    = data['category'].to(self.config.device)
    #             logits_all.append(logits.detach())
    #             labels_all.append(labels)

    #     logits_all, labels_all = merge_results_dist(logits_all, labels_all)

    #     if self.rank == 0:
    #         if logits_all is None:
    #             return

    #         dataset_size = len(self.modelnet40_loader.dataset)
    #         logits_all   = logits_all[:dataset_size]
    #         labels_all   = labels_all[:dataset_size]

    #         topk_acc, _     = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))
    #         per_cat_correct = torch.zeros(40).to(self.config.device)
    #         per_cat_count   = torch.zeros(40).to(self.config.device)

    #         for i in range(40):
    #             idx = labels_all == i
    #             if idx.sum() > 0:
    #                 per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
    #                 per_cat_count[i]   = idx.sum()

    #         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    #         per_cat_acc = per_cat_correct / per_cat_count

    #         if overall_acc > self.best_modelnet40_overall_acc:
    #             self.best_modelnet40_overall_acc = overall_acc
    #         if per_cat_acc.mean() > self.best_modelnet40_class_acc:
    #             self.best_modelnet40_class_acc = per_cat_acc.mean()

    #         logging.info(
    #             'Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(
    #                 overall_acc, self.best_modelnet40_overall_acc,
    #                 per_cat_acc.mean(), self.best_modelnet40_class_acc,
    #             )
    #         )
    #         logging.info(
    #             'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(
    #                 topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()
    #             )
    #         )
    #         torch.save(
    #             {"logits": logits_all, "labels": labels_all,
    #             "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
    #             os.path.join(self.config.ckpt_dir, f"modelnet40_epoch_{self.epoch}.pth"),
    #         )


    # def test_objaverse_lvis(self):
    #     self.model.eval()
    #     if self.config.training.use_text_proj:
    #         self.text_proj.eval()

    #     clip_text_feat = torch.from_numpy(
    #         self.objaverse_lvis_loader.dataset.clip_cat_feat
    #     ).to(self.config.device)
    #     if self.config.training.use_text_proj:
    #         clip_text_feat = self.text_proj(clip_text_feat)

    #     per_cat_correct = torch.zeros(1156).to(self.config.device)
    #     per_cat_count   = torch.zeros(1156).to(self.config.device)

    #     logits_all = []
    #     labels_all = []

    #     with torch.no_grad():
    #         for data in tqdm(self.objaverse_lvis_loader):
    #             if not self.config.model.get("use_dense", False):
    #                 pred_feat = self.model(
    #                     data['xyz'], data['features'],
    #                     device=self.config.device,
    #                     quantization_size=self.config.model.voxel_size
    #                 )
    #             else:
    #                 pred_feat = self.model(
    #                     data['xyz_dense'].to(self.config.device),
    #                     data['features_dense'].to(self.config.device)
    #                 )

    #             shape_emb = self._get_shape_embedding(pred_feat)  # (B, 1280)
    #             logits    = shape_emb @ F.normalize(clip_text_feat, dim=-1).T
    #             labels    = data['category'].to(self.config.device)
    #             logits_all.append(logits.detach())
    #             labels_all.append(labels)

    #     logits_all, labels_all = merge_results_dist(logits_all, labels_all)

    #     if self.rank == 0:
    #         if logits_all is None:
    #             return

    #         dataset_size = len(self.objaverse_lvis_loader.dataset)
    #         logits_all   = logits_all[:dataset_size]
    #         labels_all   = labels_all[:dataset_size]

    #         topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))

    #         for i in torch.unique(labels_all):
    #             idx = labels_all == i
    #             if idx.sum() > 0:
    #                 per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
    #                 per_cat_count[i]   = idx.sum()

    #         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    #         per_cat_acc = per_cat_correct / per_cat_count

    #         if overall_acc > self.best_lvis_acc:
    #             self.best_lvis_acc = overall_acc
    #             self.save_model('best_lvis')

    #         logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(
    #             overall_acc, per_cat_acc.mean()))
    #         logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(
    #             topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))

    #         torch.save(
    #             {"logits": logits_all, "labels": labels_all,
    #             "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
    #             os.path.join(self.config.ckpt_dir, f"objaverse_lvis_epoch_{self.epoch}.pth"),
    #         )


    # def test_scanobjectnn(self):
    #     self.model.eval()
    #     if self.config.training.use_text_proj:
    #         self.text_proj.eval()

    #     clip_text_feat = torch.from_numpy(
    #         self.scanobjectnn_loader.dataset.clip_cat_feat
    #     ).to(self.config.device)
    #     if self.config.training.use_text_proj:
    #         clip_text_feat = self.text_proj(clip_text_feat)

    #     per_cat_correct = torch.zeros(15).to(self.config.device)
    #     per_cat_count   = torch.zeros(15).to(self.config.device)

    #     logits_all = []
    #     labels_all = []

    #     with torch.no_grad():
    #         for data in self.scanobjectnn_loader:
    #             if not self.config.model.get("use_dense", False):
    #                 pred_feat = self.model(
    #                     data['xyz'], data['features'],
    #                     device=self.config.device,
    #                     quantization_size=self.config.model.voxel_size
    #                 )
    #             else:
    #                 pred_feat = self.model(
    #                     data['xyz_dense'].to(self.config.device),
    #                     data['features_dense'].to(self.config.device)
    #                 )

    #             shape_emb = self._get_shape_embedding(pred_feat)  # (B, 1280)
    #             logits    = shape_emb @ F.normalize(clip_text_feat, dim=-1).T
    #             labels    = data['category'].to(self.config.device)
    #             logits_all.append(logits.detach())
    #             labels_all.append(labels)

    #     logits_all, labels_all = merge_results_dist(logits_all, labels_all)

    #     if self.rank == 0:
    #         if logits_all is None:
    #             return

    #         dataset_size = len(self.scanobjectnn_loader.dataset)
    #         logits_all   = logits_all[:dataset_size]
    #         labels_all   = labels_all[:dataset_size]

    #         topk_acc, _ = self.accuracy(logits_all, labels_all, topk=(1, 3, 5))

    #         for i in range(15):
    #             idx = labels_all == i
    #             if idx.sum() > 0:
    #                 per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
    #                 per_cat_count[i]   = idx.sum()

    #         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    #         per_cat_acc = per_cat_correct / per_cat_count

    #         logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(
    #             overall_acc, per_cat_acc.mean()))
    #         logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(
    #             topk_acc[0].item(), topk_acc[1].item(), topk_acc[2].item()))

    #         torch.save(
    #             {"logits": logits_all, "labels": labels_all,
    #             "overall_acc": overall_acc, "class_acc": per_cat_acc.mean()},
    #             os.path.join(self.config.ckpt_dir, f"scanobjectnn_epoch_{self.epoch}.pth"),
    #         )   
