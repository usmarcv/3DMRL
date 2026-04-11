import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
import wandb
from numpy import *
from tqdm import tqdm
# from trainers.trainer_utils import merge_two_branch_results_dist

from collections import defaultdict


class Alignment_MRL(object):
    def __init__(self, rank, config, model, mrl_dims, logit_scale, image_proj, text_proj, optimizer,
                 scheduler, train_loader):
        
        self.rank = rank
        self.config = config

        self.model = model
        self.logit_scale = logit_scale

        self.mrl_dims = mrl_dims

        #Adapters here
        self.image_proj = image_proj
        self.text_proj = text_proj
        # self.mrl_image_proj = mrl_image_proj #for mrl alignment, we have a specific projection head for the image features that outputs the MRL dimensions, while the text_proj outputs the full CLIP dimension and is sliced in the loss function

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader

        self.epoch = 0
        self.step = 0
        # self.best_img_contras_acc = 0
        # self.best_text_contras_acc = 0
        # self.best_modelnet40_overall_acc = 0
        # self.best_modelnet40_class_acc = 0
        # self.best_lvis_acc = 0
        self.config.ngpu = dist.get_world_size()

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.config.training.use_text_proj:
            self.text_proj.load_state_dict(checkpoint['text_proj'])
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(checkpoint['image_proj'])
        if self.config.training.use_mrl_proj:
            self.mrl_image_proj.load_state_dict(checkpoint['mrl_image_proj'])
        self.logit_scale.load_state_dict(checkpoint['logit_scale'])  # module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.scheduler == "default":
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']

        #review for below
        self.best_img_contras_acc = checkpoint['best_img_contras_acc']
        self.best_text_contras_acc = checkpoint['best_text_contras_acc']
        self.best_modelnet40_overall_acc = checkpoint['best_modelnet40_overall_acc']
        self.best_modelnet40_class_acc = checkpoint['best_modelnet40_class_acc']
        self.best_lvis_acc = checkpoint['best_lvis_acc']
        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))
        logging.info("----Best img contras acc: {}".format(self.best_img_contras_acc))
        logging.info("----Best text contras acc: {}".format(self.best_text_contras_acc))
        logging.info("----Best modelnet40 overall acc: {}".format(self.best_modelnet40_overall_acc))
        logging.info("----Best modelnet40 class acc: {}".format(self.best_modelnet40_class_acc))
        logging.info("----Best lvis acc: {}".format(self.best_lvis_acc))

    def contrast_mrl_loss(self, feat1, feat2, mlr_dim, logit_scale=1, mask=None):
        
        #esses pesos na verdade são a relatie importance do mrl, investigar melhor, mas por enquanto deixamos em 1.0 para todas as dimensoes
        # weights = torch.tensor([1.0 for _ in mlr_dim], device=self.config.device) #com o 1.0 não foi testado ainda - testando
        # weights = weights / weights.sum()

        total_loss = .0
        total_acc = .0
        loss_per_dim = {}
        acc_per_dim = {}

        for i, dim_slice in enumerate(mlr_dim):
            # separando as dimensoes
            features1 = F.normalize(feat1[:, :dim_slice], dim=1)
            features2 = F.normalize(feat2[:, :dim_slice], dim=1)

            #como temos apenas uma gpu na psyduck, the code don't enter here
            if self.config.ngpu > 1:
                #concatenando os tensores de todos os gpus em cada linha da matriz 
                all_features1 = torch.cat(torch.distributed.nn.all_gather(features1), dim=0)
                all_features2 = torch.cat(torch.distributed.nn.all_gather(features2), dim=0)
                logits = logit_scale * all_features1 @ all_features2.T
                # logits = (logit_scale / math.sqrt(dim_slice)) * (all_features1 @ all_features2.T)
            else:
                logits = logit_scale * features1 @ features2.T

            if mask is not None:
                logits = logits * mask

            if mask is not None:
                if mask.dtype == torch.bool:
                    logits = logits.masked_fill(~mask, -1e9)
                else:
                    logits = logits + (1.0 - mask) * -1e9
            
            # criando labels para calcular a loss
            labels = torch.arange(logits.shape[0]).to(self.config.device)

            # calculando a loss e accuracy para a dim atual
            loss_dim_slice = (F.cross_entropy(logits, labels) 
                              + F.cross_entropy(logits.T, labels)) / 2
            
            #acc per dim sliced
            acc_dim_slice = (logits.argmax(dim=1) == labels).float().mean()

            loss_per_dim[dim_slice] = loss_dim_slice.detach().item()
            acc_per_dim[dim_slice] = acc_dim_slice.detach().item()

            #set the relative importance of each dimension slice to 1.0 for now, but can be changed later
            #the same used in MRL: https://github.com/RAIVNLab/MRL/blob/main/MRL.py#L26
            relative_importance = torch.ones_like(loss_dim_slice) 
            # Placeholder for relative importance, can be set to different values for each dimension slice

            # total_loss += weights[i] * loss_dim_slice
            total_loss += relative_importance * loss_dim_slice
            # total_acc += acc_dim_slice
            total_acc =* acc_dim_slice            

        return total_loss, total_acc, loss_per_dim, acc_per_dim


    def train_one_epoch(self):
        # Clip features extraction in evaluation mode
        self.model.train()
        # Adapters in training mode
        if self.config.training.use_text_proj:
            self.text_proj.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()
      
        epoch_loss_dims = defaultdict(list)
        epoch_acc_dims = defaultdict(list)
        mean_acc_list = []
        mean_loss_list = []
  
        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
            mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

        for data in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()

            raw_img_feat = torch.vstack(data['img_feat']).to(self.config.device)
            raw_img_feat = self.model(raw_img_feat) #forward pass through the CLIP Adapter
            # frozen model to get the features to be aligned in the mrl loss, we can also use the raw features without passing through the model, but we want to test if passing through the model improves the alignment
            raw_text_feat = torch.vstack(data['text_feat']).to(self.config.device)
                                                                   
            #indicies para alinhar imagens com textos   
            idx = data['has_text_idx']
            raw_img_feat_paired = raw_img_feat[idx]

            #Forward pass through adapters with mrl
            mrl_img_feat = self.image_proj(raw_img_feat_paired)
            mrl_text_feat = self.text_proj(raw_text_feat)

            mask = None
            if self.config.training.use_mask:
                with torch.no_grad():
                    img_text_sim = F.normalize(raw_img_feat_paired, dim=-1) @ F.normalize(raw_text_feat, dim=-1).T
                    mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                    mask = torch.logical_or(mask, mask_other).detach()

            logit_scale = self.logit_scale(None)

            loss, mean_acc, loss_dims, acc_dims = \
            self.contrast_mrl_loss(
                    mrl_img_feat, 
                    mrl_text_feat, 
                    self.mrl_dims,
                    logit_scale, 
                    mask=mask
            )
        

            loss.backward()
            self.optimizer.step()

            if self.config.training.scheduler == "cosine" or self.config.training.scheduler == "const":
                self.scheduler(self.step)
            else:
                self.scheduler.step()


            #logging from mrl dims
            #loss e acc total
            mean_acc_list.append(mean_acc.item())
            mean_loss_list.append(loss.item())
            #logging per dim
            for d, val in loss_dims.items(): epoch_loss_dims[d].append(val)
            for d, val in acc_dims.items(): epoch_acc_dims[d].append(val)

        if self.rank == 0:
            # Média geral da época
            avg_loss = np.mean(mean_loss_list) / len(self.config.mrl.dims)
            avg_acc = np.mean(mean_acc_list) / len(self.config.mrl.dims)
            
            logging.info(f"[Adapter Epoch {self.epoch}] Total Loss: {avg_loss} | Mean Acc: {avg_acc}")
            
            # Log detalhado por dimensão (Loss e Acc lado a lado)
            logging.info("-" * 100)
            logging.info(f"{'Dim '} | {'Acc'} | {'Loss'}")
            logging.info("-" * 100)
            
            for d in self.mrl_dims:
                d_acc = np.mean(epoch_acc_dims[d])
                d_loss = np.mean(epoch_loss_dims[d])
                logging.info(f"D{d} | {d_acc}     | {d_loss}")
            
            logging.info("-" * 100)


    def save_model(self, name):
        torch.save({
            "state_dict": self.model.state_dict(),
            "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
            "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
            "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.scheduler == "default" else None,
            "epoch": self.epoch,
            "step": self.step,
        }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))


    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            
            if self.rank == 0:
                logging.info("Alignment MRL Training Epoch: {}".format(self.epoch))
            self.train_one_epoch()

            if self.rank == 0:
                self.save_model('latest')
                if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                    self.save_model('epoch_{}'.format(self.epoch))