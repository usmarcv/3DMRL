import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from tqdm import tqdm
from trainers.trainer_utils import merge_results_dist

class FSL_Trainer(object):
    def __init__(self, rank, config, model, mrl_heads, linear_layer, optimizer,
                 scheduler, train_loader, test_loader, image_branch=None, text_branch=None):
        self.rank = rank
        self.config = config
        self.model = model
        self.mrl_heads = mrl_heads
        self.linear_layer = linear_layer

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch = 0
        self.step = 0

        self.config.ngpu = dist.get_world_size()
        self.image_branch = image_branch
        self.text_branch = text_branch
        self.criterion = nn.CrossEntropyLoss()
        
        # MRL Adaptation: Initialize dictionaries for tracking accuracy per dimension
        self.mrl_dims = [str(dim) for dim in self.config.linear_layer.mrl_dims]
        self.best_overall_acc = {dim: 0.0 for dim in self.mrl_dims}
        self.best_class_acc = {dim: 0.0 for dim in self.mrl_dims}

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        # Atenção: se for retomar o linear probing, você deve carregar os pesos da linear_layer aqui
        self.linear_layer.load_state_dict(checkpoint['mrl_heads'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.scheduler == "default":
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

    def train_one_epoch(self):
        # 1. Congelar Backbone e MRL Heads, treinar apenas Linear Layer
        self.model.eval()
        self.mrl_heads.eval()
        self.linear_layer.train()
        
        if self.image_branch is not None and self.text_branch is not None:
            self.image_branch.eval()
            self.text_branch.eval()

        # --- TRAINING LOOP ---
        for data in tqdm(self.train_loader, desc=f"Train Epoch {self.epoch}" if self.rank == 0 else None):
            self.step += 1
            self.optimizer.zero_grad()
            
            # Tudo que é pré-treinado fica no bloco no_grad() para economizar RAM
            with torch.no_grad():
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                
                # CORREÇÃO 1: Se o backbone (model) retornar uma tupla, pegue o primeiro elemento
                if isinstance(pred_feat, tuple):
                    pred_feat = pred_feat[0]

                # Passar a feature bruta pelas MRL Heads
                mrl_feat = self.mrl_heads(pred_feat)
                
                # CORREÇÃO 2: Se a MRL Head retornar uma tupla, pegue o primeiro elemento
                if isinstance(mrl_feat, tuple):
                    mrl_feat = mrl_feat[0]
                
            # A camada linear recebe as features já projetadas no espaço MRL (agora garantidamente um Tensor)
            outputs = self.linear_layer(mrl_feat)
            label = data["category"].to(self.config.device)
            
            # Sum the loss for all MRL dimensions using Python's native sum()
            loss = sum([self.criterion(outputs[dim], label.long()) for dim in outputs.keys()])
            loss.backward()

            self.optimizer.step()
            if self.config.training.scheduler in ["cosine", "const"]:
                self.scheduler(self.step)
            else:
                self.scheduler.step()

        # --- TESTING LOOP ---
        num_cates = self.config.dataset.NUM_CATEGORY
        
        self.model.eval()
        self.mrl_heads.eval()
        self.linear_layer.eval()
        
        # MRL Adaptation: Dictionary to hold logits for each dimension
        logits_all = {dim: [] for dim in self.mrl_dims}
        labels_all = []
        
        with torch.no_grad():
            for data in tqdm(self.test_loader, desc=f"Test Epoch {self.epoch}" if self.rank == 0 else None):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                            device=self.config.device, \
                                            quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                # CORREÇÃO 1: Tratar retorno do Backbone
                if isinstance(pred_feat, tuple):
                    pred_feat = pred_feat[0]

                # Passar a feature bruta pelas MRL Heads
                mrl_feat = self.mrl_heads(pred_feat)

                # CORREÇÃO 2: Tratar retorno das MRL Heads
                if isinstance(mrl_feat, tuple):
                    mrl_feat = mrl_feat[0]

                labels = data['category'].to(self.config.device)
                labels_all.append(labels)

                # Get the multi-head outputs and store them independently
                outputs = self.linear_layer(mrl_feat)
                for dim_str in outputs.keys():
                    logits_all[str(dim_str)].append(outputs[dim_str])

        # --- DISTRIBUTED EVALUATION AND METRICS PER MRL DIMENSION ---
        table_rows = []
        dataset_name = self.config.dataset.NAME 
        current_fold = self.config.dataset.fold
        
        for dim_str in self.mrl_dims:
            # We use list(labels_all) to pass a copy, so merge_results_dist doesn't empty the original list
            merged_logits, merged_labels = merge_results_dist(logits_all[dim_str], list(labels_all))
            
            if self.rank == 0:
                topk_acc, _ = self.accuracy(merged_logits, merged_labels, topk=(1, 3, 5,))
                
                per_cat_correct = torch.zeros(num_cates).cuda()
                per_cat_count = torch.zeros(num_cates).cuda()

                for i in range(num_cates):
                    idx = (merged_labels == i)
                    if idx.sum() > 0:
                        per_cat_correct[i] = (merged_logits[idx].argmax(dim=1) == merged_labels[idx]).float().sum()
                        per_cat_count[i] = idx.sum()

                overall_acc_val = (per_cat_correct.sum() / per_cat_count.sum()).item()
                class_acc_val = (per_cat_correct / per_cat_count).mean().item()
                valid_classes = per_cat_count > 0
                if valid_classes.sum() > 0:
                    class_acc_val = (per_cat_correct[valid_classes] / per_cat_count[valid_classes]).mean().item()
                else:
                    class_acc_val = 0.0
                    
                # Update best overall accuracy and save model for this dimension
                if overall_acc_val > self.best_overall_acc[dim_str]:
                    self.best_overall_acc[dim_str] = overall_acc_val
                    self.save_model(f'best_{dataset_name}_fold{current_fold}_overall_dim_{dim_str}')
                    
                if class_acc_val > self.best_class_acc[dim_str]:
                    self.best_class_acc[dim_str] = class_acc_val
                    self.save_model(f'best_{dataset_name}_fold{current_fold}_class_dim_{dim_str}')

                best_oa = self.best_overall_acc[dim_str] * 100
                best_ca = self.best_class_acc[dim_str] * 100

                row = f"| {dim_str:>6} | {overall_acc_val*100:>6.2f}% ({best_oa:>6.2f}%) | {class_acc_val*100:>6.2f}% ({best_ca:>6.2f}%) | {topk_acc[0].item():>6.2f}% | {topk_acc[1].item():>6.2f}% | {topk_acc[2].item():>6.2f}% |"
                table_rows.append(row)

        if self.rank == 0:
            header = f"| {'Dim':>6} | {'Overall Acc (Best)':>17} | {'Class Acc (Best)':>17} | {'Top-1':>7} | {'Top-3':>7} | {'Top-5':>7} |"
            separator = "-" * len(header)
            table_str = "\n" + separator + "\n" + header + "\n" + separator + "\n" + "\n".join(table_rows) + "\n" + separator
            
            logging.info(f"Test Results Epoch {self.epoch} (Fold {current_fold}):{table_str}")

    def save_model(self, name):
        torch.save({
            # Salvando APENAS a linear layer (Linear Probing), o resto já está salvo no checkpoint original
            "state_dict": self.linear_layer.state_dict(),
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
        current_fold = self.config.dataset.fold
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info(f"Fold: {current_fold} | Epoch: {self.epoch}")
                
            self.train_one_epoch()
            
            if self.rank == 0:
                self.save_model(f'latest_fold{current_fold}')
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model(f'epoch_{self.epoch}_fold{current_fold}')
                
        return self.best_overall_acc, self.best_class_acc