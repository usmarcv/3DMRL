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

class CLIP_Adapter_Trainer(object):
    def __init__(self, rank, config, model, mrl_dims, logit_scale, image_proj, text_proj, optimizer,
                 scheduler, train_loader):
        self.rank = rank
        self.config = config
        self.model = model
        self.mrl_dims = mrl_dims
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.epoch = 0
        self.step = 0
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0
        self.config.ngpu = dist.get_world_size()

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.config.training.use_text_proj:
            self.text_proj.load_state_dict(checkpoint['text_proj'])
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(checkpoint['image_proj'])

        self.logit_scale.load_state_dict(checkpoint['logit_scale'])  # module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.scheduler == "default":
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
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

    def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            # i=5
            # if i<4:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
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
        # print(logits.argmax(dim=1))
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

    def compute_logit(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            # i=5
            # if i<4:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
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
        # print(logits.argmax(dim=1))
        # loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return logits

        
    def contrast_mrl_loss(self, feat1, feat2, mlr_dim, logit_scale=1, mask=None):
        
        if self.config.mrl.use_mrl_log_scale:
            list_weights = torch.tensor([1.0 / math.log2(dim) for dim in mlr_dim]).to(self.config.device)
            relative_importance = list_weights / list_weights.mean()

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
                logits = logit_scale * (all_features1 @ all_features2.T)
                # logits = (logit_scale / math.sqrt(dim_slice)) * (all_features1 @ all_features2.T)
            else:
                logits = logit_scale * (features1 @ features2.T)

            # if mask is not None:
            #     logits = logits * mask

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


            # total_loss += weights[i] * loss_dim_slice
            if self.config.mrl.use_mrl_log_scale:
                total_loss += relative_importance[i] * loss_dim_slice
                total_acc += relative_importance[i] * acc_dim_slice    
            else: 
                relative_importance = torch.ones_like(loss_dim_slice) 
                total_loss += relative_importance * loss_dim_slice
                total_acc += relative_importance * acc_dim_slice       

        return total_loss, total_acc, loss_per_dim, acc_per_dim
 


    def train_one_epoch(self):
        self.model.eval()
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
        img_text_pair_before = {}
        img_text_pair_after = {}
        for data in tqdm(self.train_loader):
            # print(data)
            self.step += 1
            self.optimizer.zero_grad()
            text_feat = torch.vstack(data['text_feat']).to(self.config.device)
            img_feat = torch.vstack(data['img_feat']).to(self.config.device)
            # name = data["name"]
            # group = data["group"]
            # texts = data["texts"]
            idx = data['has_text_idx']
            image_idx = data["image_idx"]
            # print(image_idx)

            # logit_scale = self.logit_scale(None)
            # logits = self.compute_logit(img_feat[idx], text_feat, logit_scale=logit_scale,
            #                                               mask=None)

            # labels = torch.arange(logits.shape[0]).to(self.config.device)
            # # print("before", (logits.argmax(dim=1) == labels))
            # logit_result = logits.argmax(dim=1)

            # acc_before = logits.argmax(dim=1) == labels


            img_feat = self.model(img_feat)

            # logits = self.compute_logit(img_feat[idx], text_feat, logit_scale=logit_scale,
            #                             mask=None)
            # labels = torch.arange(logits.shape[0]).to(self.config.device)
            # # print("after", (logits.argmax(dim=1) == labels))
            # acc_after = logits.argmax(dim=1) == labels


            # for i in range(len(acc_after)):
            #     if acc_before[i] == False and acc_after[i] == True:
            #         # print(group[i])
            #         # print(image_idx[i])
            #         group_id = group[i]
            #         idx = image_idx[i]
            #         object_name = name[i]

            #         text = texts[logit_result[i]]
            #         image_path = os.path.join(group_id, object_name, "colors_" + str(idx) + ".png")

            #         img_text_pair_before[image_path] = text

            #         img_text_pair_after[image_path] = texts[i]


            # print(img_feat[idx].shape, "image feat")
            # print(text_feat[idx].shape, "text feat")

            logit_scale = self.logit_scale(None)
            
            if self.config.training.use_mask:
                img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
                mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                mask = torch.logical_or(mask, mask_other).detach()
            else:
                mask = None
            
            if self.config.training.use_image_proj:
                img_feat = self.image_proj(img_feat)
            
            if self.config.training.use_text_proj:
                text_feat = self.text_proj(text_feat)
            
            
            loss = 0.0
            
            
            loss, mean_acc, loss_dims, acc_dims = self.contrast_mrl_loss(
                                                            img_feat[idx], 
                                                            text_feat,
                                                            self.mrl_dims, 
                                                            logit_scale=logit_scale,
                                                            mask=mask
                                                            )
            
            
            # loss += contras_loss
            # text_contras_acc_list.append(contras_acc.item())
            # print("here, ", self.rank)

            mean_acc_list.append(mean_acc.item())
            mean_loss_list.append(loss.item())
            #logging per dim
            for d, val in loss_dims.items(): epoch_loss_dims[d].append(val)
            for d, val in acc_dims.items(): epoch_acc_dims[d].append(val)

            
            loss.backward()
            self.optimizer.step()

            if self.config.training.scheduler == "cosine" or self.config.training.scheduler == "const":
                self.scheduler(self.step)
            else:
                self.scheduler.step()
            
            
        if self.rank == 0:
            avg_loss = np.mean(mean_loss_list)
            avg_acc = np.mean(mean_acc_list)

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

            
        # if self.rank == 0:
        #     logging.info('Train: contrast acc: {0}' \
        #                  .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0))


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
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()
            if self.rank == 0:
                self.save_model('latest')
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))