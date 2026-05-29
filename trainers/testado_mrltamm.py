import logging
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

class CLIP_Adapter_Trainer(object):
    def __init__(self, rank, config, image_adapter, text_adapter, logit_scale, image_proj, text_proj, optimizer,
                 scheduler, train_loader):
        self.rank = rank
        self.config = config
        
        # Adaptadores Principais (Residuais)
        self.image_adapter = image_adapter 
        self.text_adapter = text_adapter 
        
        # Projeções Lineares Extras
        self.image_proj = image_proj
        self.text_proj = text_proj
        
        self.logit_scale = logit_scale
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        
        self.epoch = 0
        self.step = 0
        self.config.ngpu = dist.get_world_size() if dist.is_initialized() else 1

        # Variáveis de controle de métricas
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        
        self.image_adapter.load_state_dict(checkpoint['image_adapter'])
        self.text_adapter.load_state_dict(checkpoint['text_adapter'])
            
        if self.config.training.use_image_proj and self.image_proj is not None:
            self.image_proj.load_state_dict(checkpoint['image_proj'])
            
        if self.config.training.get('use_text_proj', False) and self.text_proj is not None:
            if checkpoint.get('text_proj') is not None:
                self.text_proj.load_state_dict(checkpoint['text_proj'])

        self.logit_scale.load_state_dict(checkpoint['logit_scale'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        if self.config.training.scheduler == "default" and checkpoint.get('scheduler') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        logging.info("Loaded checkpoint from {}".format(path))

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

    # def train_one_epoch(self):
    #     # 1. Coloca tudo em modo de treino
    #     self.image_adapter.train()
    #     self.text_adapter.train()
        
    #     if self.config.training.use_image_proj:
    #         self.image_proj.train()
    #     if self.config.training.use_text_proj:
    #         self.text_proj.train()

    #     mrl_dims = self.config.nesting_list
    #     text_contras_acc_list = [] # to avg at the end of the epoch

    #     dim_acc_tracker = {dim: [] for dim in mrl_dims}
    #     dim_loss_tracker = {dim: [] for dim in mrl_dims}
        
    #     # Máscara de hard negatives
    #     if self.config.training.use_mask:
    #         k = self.config.dataset.negative_sample_num
    #         s = self.config.dataset.train_batch_size
    #         mask1 = np.eye(k * s).astype(np.bool_)
    #         mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool_)
    #         mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

    #     for data in tqdm(self.train_loader):
    #         self.step += 1
    #         self.optimizer.zero_grad()
            
    #         text_feat = torch.vstack(data['text_feat']).to(self.config.device)
    #         img_feat = torch.vstack(data['img_feat']).to(self.config.device)
    #         idx = data['has_text_idx']
            
    #         # 2. Passagem 1: Adaptadores NewCLIP (Residuais)
    #         img_feat_adapted = self.image_adapter(img_feat)
    #         text_feat_adapted = self.text_adapter(text_feat)
            
    #         # 3. Passagem 2: Projeções Lineares Extras (Opcionais no config)
    #         if self.config.training.use_image_proj:
    #             img_feat_adapted = self.image_proj(img_feat_adapted)
                
    #         if self.config.training.use_text_proj:
    #             text_feat_adapted = self.text_proj(text_feat_adapted)

    #         # 4. Máscara (usando as features de dimensão máxima 1024)
    #         mask = None
    #         if self.config.training.use_mask:
    #             img_text_sim = F.normalize(text_feat_adapted, dim=-1) @ F.normalize(img_feat_adapted[idx], dim=-1).T
    #             mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
    #             mask = torch.logical_or(mask, mask_other).detach()

    #         # 5. O CORAÇÃO DO MRL: Loop pelas dimensões
    #         logit_scale = self.logit_scale(None)
    #         total_loss = 0.0
    #         avg_acc = 0.0
            
    #         for dim in mrl_dims:
    #             # Fatiamento dinâmico
    #             img_slice = img_feat_adapted[idx, :dim]
    #             text_slice = text_feat_adapted[idx:, :dim]
                
    #             slice_loss, slice_acc = self.contras_loss(
    #                 img_slice, 
    #                 text_slice, 
    #                 logit_scale=logit_scale,
    #                 mask=mask
    #             )
                
    #             total_loss += slice_loss
    #             avg_acc += slice_acc

    #             dim_loss_tracker[dim].append(slice_loss.item())
    #             dim_acc_tracker[dim].append(slice_acc.item())
                
    #         avg_acc = avg_acc / len(mrl_dims)
    #         text_contras_acc_list.append(avg_acc.item())

    #         # 6. Otimização
    #         total_loss.backward()
    #         self.optimizer.step()
            
    #         if self.config.training.scheduler in ["cosine", "const"]:
    #             self.scheduler(self.step)
    #         else:
    #             self.scheduler.step()

    #     if self.rank == 0:
    #         logging.info(f'--- Resumo da Época {self.epoch} ---')
    #         for dim in mrl_dims:
    #             avg_dim_loss = np.mean(dim_loss_tracker[dim])
    #             avg_dim_acc = np.mean(dim_acc_tracker[dim])
    #             logging.info(f'Dim {dim:>4}: Loss = {avg_dim_loss:.4f} | Acc = {avg_dim_acc:.4f}')
            
    #         logging.info('  -----------------------------------------')
    #         logging.info('  MRL Avg Acc: {0:.4f}'.format(
    #             np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0))
    #         logging.info('=============================================')


  

    def train_one_epoch(self):
        # 1. Coloca tudo em modo de treino
        self.image_adapter.train()
        self.text_adapter.train()
        
        if self.config.training.use_image_proj:
            self.image_proj.train()
        if self.config.training.use_text_proj:
            self.text_proj.train()

        mrl_dims = self.config.nesting_list
        dim_acc_tracker = {dim: [] for dim in mrl_dims}
        dim_loss_tracker = {dim: [] for dim in mrl_dims}
        text_contras_acc_list = []

        for data in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            
            text_feat = torch.vstack(data['text_feat']).to(self.config.device)
            img_feat = torch.vstack(data['img_feat']).to(self.config.device)
            idx = data['has_text_idx']
            
            # 2. Passagem 1: Adaptadores NewCLIP
            img_feat_adapted = self.image_adapter(img_feat)
            text_feat_adapted = self.text_adapter(text_feat)
            
            # 3. Passagem 2: Projeções Lineares
            if self.config.training.use_image_proj:
                img_feat_adapted = self.image_proj(img_feat_adapted)
                
            if self.config.training.get('use_text_proj', False):
                text_feat_adapted = self.text_proj(text_feat_adapted)

            # --- CORREÇÃO 3: Filtramos os features válidos primeiro para saber o tamanho real do lote ---
            img_feat_valid = img_feat_adapted[idx]
            text_feat_valid = text_feat_adapted[idx]
            current_batch_size = img_feat_valid.shape[0] # Tamanho dinâmico (M)

            mask = None
            if self.config.training.use_mask and current_batch_size > 0:
                # Recalcula a mask_other para o tamanho exato deste lote filtrado
                k = self.config.dataset.negative_sample_num
                mask1 = np.eye(k * current_batch_size).astype(np.bool_)
                mask2 = np.kron(np.eye(current_batch_size), np.ones((k, k))).astype(np.bool_)
                mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

                # CORREÇÃO 2: Usa ambos os tensores já filtrados para manter a matriz quadrada (MxM)
                img_text_sim = F.normalize(text_feat_valid, dim=-1) @ F.normalize(img_feat_valid, dim=-1).T
                mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                mask = torch.logical_or(mask, mask_other).detach()

            # 5. O CORAÇÃO DO MRL: Loop pelas dimensões
            logit_scale = self.logit_scale(None)
            total_loss = 0.0
            avg_acc = 0.0
            
            # Só processa a loss se houver itens válidos no batch
            if current_batch_size > 0:
                for dim in mrl_dims:
                    # CORREÇÃO 1: Fatiamento dinâmico sem o "idx:"
                    img_slice = img_feat_valid[:, :dim]
                    text_slice = text_feat_valid[:, :dim]
                    
                    slice_loss, slice_acc = self.contras_loss(
                        img_slice, 
                        text_slice, 
                        logit_scale=logit_scale,
                        mask=mask
                    )
                    
                    total_loss += slice_loss
                    avg_acc += slice_acc

                    dim_loss_tracker[dim].append(slice_loss.item())
                    dim_acc_tracker[dim].append(slice_acc.item())
                    
                avg_acc = avg_acc / len(mrl_dims)
                text_contras_acc_list.append(avg_acc.item())

                # 6. Otimização
                total_loss.backward()
                self.optimizer.step()
            
            # Scheduler avança independente de ter tido texto válido no batch ou não
            if self.config.training.scheduler in ["cosine", "const"]:
                self.scheduler(self.step)
            else:
                self.scheduler.step()


        # Resumo final da época (mantido igual)
        if self.rank == 0:
            logging.info(f'--- Resumo da Época {self.epoch} ---')
            for dim in mrl_dims:
                # Evita avisos do numpy se a lista estiver vazia
                if len(dim_loss_tracker[dim]) > 0:
                    avg_dim_loss = np.mean(dim_loss_tracker[dim])
                    avg_dim_acc = np.mean(dim_acc_tracker[dim])
                    logging.info(f'Dim {dim:>4}: Loss = {avg_dim_loss:.4f} | Acc = {avg_dim_acc:.4f}')
            
            logging.info('  -----------------------------------------')
            logging.info('  MRL Avg Acc: {0:.4f}'.format(
                np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0))
            logging.info('=============================================')


    def save_model(self, name):
        torch.save({
            "image_adapter": self.image_adapter.state_dict(),
            "text_adapter": self.text_adapter.state_dict(),
            "logit_scale": self.logit_scale.state_dict(),
            "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
            "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.scheduler == "default" else None,
            "epoch": self.epoch,
            "step": self.step,
        }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))
        
        
    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()

            if self.rank == 0:
                self.save_model('latest')
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))