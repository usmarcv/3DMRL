import logging
import os

import numpy as np
import torch
import torch.distributed.nn
import torch.nn.functional as F
import wandb
import torch.distributed as dist
# from trainers.trainer_utils import merge_results_dist
from tqdm import tqdm
from trainers.trainer_utils import merge_two_branch_results_dist, merge_results_dist

# from models.hierarchical_mrl_head import Matryoskha_Constrastive_Loss, Hierarchical_MRL_Head

import math
from collections import OrderedDict, defaultdict

class Trainer_MRL(object):
    def __init__(self, rank, config, model, logit_scale,
                image_proj, text_proj, mrl_image_proj, mrl_text_proj, optimizer, 
                scheduler, train_loader, image_alignment_adapter, text_alignment_adapter,
                modelnet40_loader, objaverse_lvis_loader, scanobjectnn_loader):

        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.mrl_image_proj = mrl_image_proj
        self.mrl_text_proj = mrl_text_proj
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        
        self.text_alignment_adapter = text_alignment_adapter
        self.image_alignment_adapter = image_alignment_adapter

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
        self.config.ngpu = dist.get_world_size()


    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])

        self.image_alignment_adapter.load_state_dict(checkpoint['image_alignment_adapter'])
        self.text_alignment_adapter.load_state_dict(checkpoint['text_alignment_adapter'])

        self.logit_scale.load_state_dict(checkpoint['logit_scale'])  # module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.use_openclip_optimizer_scheduler == False:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] 
        self.step = checkpoint['step']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

    
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
            loss_dim_slice = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            
            #acc per dim sliced
            acc_dim_slice = (logits.argmax(dim=1) == labels).float().mean()

            loss_per_dim[dim_slice] = loss_dim_slice.detach().item()
            acc_per_dim[dim_slice] = acc_dim_slice.detach().item()


            # total_loss += weights[i] * loss_dim_slice
            if self.config.mrl.use_mrl_log_scale:
                total_loss += relative_importance[i] * loss_dim_slice
                total_acc += acc_dim_slice    
            else: 
                # relative_importance = torch.ones(len(mlr_dim))
                relative_importance = torch.ones_like(loss_dim_slice)
                total_loss += relative_importance * loss_dim_slice
                total_acc += acc_dim_slice       

        return total_loss, total_acc/len(mlr_dim), loss_per_dim, acc_per_dim
 

   

    def train_one_epoch(self):
        
        self.model.train()
        if self.config.training.use_text_proj: #True
            self.text_proj.train() 
        if self.config.training.use_image_proj: 
            self.image_proj.train()


        # clip-adapter layers here to improve image feature and text feature
        self.image_alignment_adapter.train() 
        self.text_alignment_adapter.train()

        text_contras_acc_list = []
        img_contras_acc_list = []

        # MRL metrics por dimensão
        epoch_img_loss_dim = defaultdict(list)
        epoch_img_acc_dim = defaultdict(list)
        epoch_txt_loss_dim = defaultdict(list)
        epoch_txt_acc_dim = defaultdict(list)

        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
            mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

        for data in tqdm(self.train_loader):
            self.step += 1
            self.optimizer.zero_grad()
            loss = 0.0
            #3D backbone forward
            if not self.config.model.get("use_dense", False):
                pred_feat = self.model(data['xyz'], data['features'], device=self.config.device,
                                       quantization_size=self.config.model.voxel_size)
            else:
                pred_feat = self.model(data['xyz_dense'].to(self.config.device), \
                                       data['features_dense'].to(self.config.device))

            logit_scale = self.logit_scale(None)
            
            #collect the image and text features
            idx = data['has_text_idx']
            text_feat = torch.vstack(data['text_feat']).to(self.config.device)
            img_feat = torch.vstack(data['img_feat']).to(self.config.device)

            if self.config.training.use_mask:
                img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
                mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                mask = torch.logical_or(mask, mask_other).detach()
            else:
                mask = None

            if self.config.training.mlp_type is not None:
                pc_image_feat = self.image_alignment_adapter(pred_feat)
                pc_text_feat = self.text_alignment_adapter(pred_feat)

            #feat number of images

            if self.config.training.loss_type == "two_branch":
                for i in range(self.config.dataset.num_imgs):
                    single_image_feat = img_feat[ : , i * self.config.clip_embed_dim : (i + 1) * self.config.clip_embed_dim]
                    single_image_feat = img_feat.to(self.config.device)

                    #projection option
                    #na verdade aqui esta fazendo a mesma coisa, eu poderia fazer utilizando a infonce classica
                    #futuramente aprumar isso aqui (JOINHA)
                    if self.config.training.use_image_proj:
                        single_image_feat = self.image_proj(single_image_feat)

                    if self.config.training.use_mrl_image_proj:
                        single_image_feat = self.mrl_image_proj(single_image_feat)

                    img_contras_loss, img_contras_acc, \
                    img_loss_dims, img_acc_dims =  self.contrast_mrl_loss(pc_image_feat, single_image_feat,
                                                       self.config.mrl.dims,
                                                       logit_scale=logit_scale,
                                                       mask=mask)
                    
                    loss += img_contras_loss * self.config.training.lambda_img_contras
                    img_contras_acc_list.append(img_contras_acc.item())
                 

                for d, val in img_loss_dims.items():
                    epoch_img_loss_dim[d].append(val)
                for d, val in img_acc_dims.items():
                    epoch_img_acc_dim[d].append(val)


                #feat number of texts
                if len(idx) > 0:
                    if self.config.training.use_text_proj:
                        text_feat = self.text_proj(text_feat)

                    if self.config.training.use_mrl_text_proj:
                        text_feat = self.mrl_text_proj(text_feat)

                    text_contras_loss, text_contras_acc, \
                    txt_loss_dims, txt_acc_dims = self.contrast_mrl_loss(pc_text_feat[idx], text_feat,
                                                    self.config.mrl.dims,
                                                    logit_scale=logit_scale, mask=mask)

                    loss += text_contras_loss * self.config.training.lambda_text_contras
                    text_contras_acc_list.append(text_contras_acc.item())


                for d, val in txt_loss_dims.items():
                    epoch_txt_loss_dim[d].append(val)
                for d, val in txt_acc_dims.items():
                    epoch_txt_acc_dim[d].append(val)
                   
                   

                # text_contras_acc_list.append(text_contras_acc.item())

            loss.backward()
            self.optimizer.step()

            if self.config.training.scheduler == "cosine" or self.config.training.scheduler == "const":
                self.scheduler(self.step)
            else:
                self.scheduler.step()

        if self.rank == 0:
            logging.info('Train avg: text_constrat_acc: {0} image_contrast_acc: {1}' \
                         .format((np.mean(text_contras_acc_list)),
                                 (np.mean(img_contras_acc_list)  if len(img_contras_acc_list) > 0 else 0)))

                    
            header = f"{'Dim':<6} | {'Img Loss':<10} | {'Img Acc':<10} | {'Txt Loss':<10} | {'Txt Acc':<10}"
            logging.info("-" * len(header))
            logging.info(header)
            logging.info("-" * len(header))
            
            all_dims = sorted(set(epoch_img_loss_dim.keys()) | set(epoch_txt_loss_dim.keys()))
            
            for d in all_dims:
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
            "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
            "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
            "optimizer": self.optimizer.state_dict(),
            "image_alignment_adapter": self.image_alignment_adapter.state_dict(),
            "text_alignment_adapter": self.text_alignment_adapter.state_dict(),
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


    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()
            if epoch > self.config.training.test_epoch:
                self.test_modelnet40()
                self.test_objaverse_lvis()
                self.test_scanobjectnn()
            if self.rank == 0:
                self.save_model('latest')
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))


    def test_modelnet40(self):

        self.model.eval()
        self.text_alignment_adapter.eval()
        self.image_alignment_adapter.eval()

        if self.config.training.use_text_proj:
            self.text_proj.eval()

        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).cuda()
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

        per_cat_correct = torch.zeros(40).cuda()
        per_cat_count = torch.zeros(40).cuda()
        category2idx = self.modelnet40_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_image_all = []
        logits_text_all = []
        labels_all = []
        with torch.no_grad():
            for data in tqdm(self.modelnet40_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'].to(self.config.device), data['features_dense'].to(self.config.device))

                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
                pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)

                logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
                logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T

                labels = data['category'].to(self.config.device)
                logits_image_all.append(logits_image.detach())
                logits_text_all.append(logits_text.detach())
                labels_all.append(labels)

        # logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
        #     os.path.join(self.config.ckpt_dir, "modelnet40_dir"),
        #     logits_image_all, logits_text_all, labels_all)
        logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
                                                            logits_image_all,
                                                            logits_text_all,
                                                            labels_all)


        if self.rank == 0:
            logits_all = logits_text_all + self.alpha * logits_image_all
            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            for i in range(40):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            if overall_acc > self.best_modelnet40_overall_acc:
                self.best_modelnet40_overall_acc = overall_acc
                self.save_model('best_modelnet40_overall')
            if per_cat_acc.mean() > self.best_modelnet40_class_acc:
                self.best_modelnet40_class_acc = per_cat_acc.mean()
                self.save_model('best_modelnet40_class')

            logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
                                                                                             self.best_modelnet40_overall_acc,
                                                                                             per_cat_acc.mean(),
                                                                                             self.best_modelnet40_class_acc))
            logging.info(
                'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                    topk_acc[1].item(),
                                                                                     topk_acc[2].item()))
            torch.save({
                        "logits_image_all": logits_image_all,
                        "logits_text_all": logits_text_all,
                        "labels_all": labels_all,
                        "overall_acc": overall_acc,
                        "class_acc": per_cat_acc.mean()
                         },
                        os.path.join(self.config.ckpt_dir,f"modelnet40_epoch_{self.epoch}.pth"))             



    def test_objaverse_lvis(self):
        self.model.eval()
        self.text_alignment_adapter.eval()
        self.image_alignment_adapter.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).cuda()
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(1156).cuda()
        per_cat_count = torch.zeros(1156).cuda()
        category2idx = self.objaverse_lvis_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_image_all = []
        logits_text_all = []
        labels_all = []
        with torch.no_grad():
            for data in tqdm(self.objaverse_lvis_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
                pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)
                # print("pred_feat_text", pred_feat_text.shape)
                # print("clip_text_feat", clip_text_feat.shape)
                logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
                logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_image_all.append(logits_image.detach())
                logits_text_all.append(logits_text.detach())
                labels_all.append(labels)

        logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
                                                            logits_image_all,
                                                            logits_text_all,
                                                            labels_all)


        if self.rank == 0:

            logits_all = 1 * logits_text_all + self.alpha * logits_image_all
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
                self.save_model('best_lvis')

            logging.info(
                'Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
            logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                                topk_acc[1].item(),
                                                                                                topk_acc[2].item()))
            torch.save({
                        "logits_image_all": logits_image_all,
                        "logits_text_all": logits_text_all,
                        "labels_all": labels_all,
                        "overall_acc": overall_acc,
                        "class_acc": per_cat_acc.mean()
                         },
                        os.path.join(self.config.ckpt_dir,f"objaverse_lvis_epoch_{self.epoch}.pth"))             


    def test_scanobjectnn(self):
        self.model.eval()
        self.text_alignment_adapter.eval()
        self.image_alignment_adapter.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count = torch.zeros(15).to(self.config.device)
        category2idx = self.scanobjectnn_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_image_all = []
        logits_text_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.scanobjectnn_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
                pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)

                logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
                logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T

                labels = data['category'].to(self.config.device)
                logits_image_all.append(logits_image.detach())
                logits_text_all.append(logits_text.detach())
                labels_all.append(labels)

        logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
                                                            logits_image_all,
                                                            logits_text_all,
                                                            labels_all)

        if self.rank == 0:

            logits_all = 1 * logits_text_all + self.alpha * logits_image_all

            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            # calculate per class accuracy
            for i in range(15):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            logging.info(
                'Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
            logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                               topk_acc[1].item(),
                                                                                               topk_acc[2].item()))


            torch.save({
                        "logits_image_all": logits_image_all,
                        "logits_text_all": logits_text_all,
                        "labels_all": labels_all,
                        "overall_acc": overall_acc,
                        "class_acc": per_cat_acc.mean()
                         },
                        os.path.join(self.config.ckpt_dir,f"scanobjectnn_epoch_{self.epoch}.pth"))             




    # def test_modelnet40(self):
    #     self.model.eval()
    #     self.mrl_image_proj.eval() if self.config.training.use_mrl_image_proj else None
    #     self.mrl_text_proj.eval() if self.config.training.use_mrl_text_proj else None
    #     if self.config.training.use_text_proj:
    #         self.text_proj.eval()

    #     clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).to(self.config.device)
    #     if self.config.training.use_text_proj:
    #         clip_text_feat = self.text_proj(clip_text_feat)

    #     category2idx = self.modelnet40_loader.dataset.category2idx
    #     idx2category = {v: k for k, v in category2idx.items()}

    #     logits_image_all = []
    #     logits_text_all = []
    #     logits_all = []
    #     labels_all = []
    #     with torch.no_grad():
    #         for data in self.modelnet40_loader:
    #             if not self.config.model.get("use_dense", False):
    #                 pred_feat = self.model(data['xyz'], data['features'], \
    #                                        device=self.config.device, \
    #                                        quantization_size=self.config.model.voxel_size)
    #             else:
    #                 pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                
    #             labels = data['category'].to(self.config.device)

    #             current_logits_image = None
    #             current_logits_text = None

    #             if self.config.training.use_mrl_image_proj:
    #                 pred_feat_img = F.normalize(self.mrl_image_proj(pred_feat), dim=1)
    #                 current_logits_image = pred_feat_img @ F.normalize(clip_text_feat, dim=1).T


    #             if self.config.training.use_mrl_text_proj:
    #                 pred_feat_text = F.normalize(self.mrl_text_proj(clip_text_feat), dim=1)
    #                 current_logits_text = F.normalize(pred_feat, dim=1) @ pred_feat_text.T
              
    #             if current_logits_image is not None and current_logits_text is not None:
    #                 # final_logits = logits_text_all + self.alpha * logits_image_all
    #                 final_logits = (self.alpha * current_logits_image) + ((1 - self.alpha) * current_logits_text)
    #             elif current_logits_image is not None:
    #                 final_logits = current_logits_image
    #             elif current_logits_text is not None:
    #                 final_logits = current_logits_text
    #             else:
    #                 final_logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T

    #             logits_all.append(final_logits.detach())
    #             labels_all.append(labels)


    #     logits_all, labels_all = merge_results_dist(logits_all, labels_all)


    #     if self.rank == 0:

    #         dataset_size = len(self.modelnet40_loader.dataset)
    #         # logits_all = logits_text_all + self.alpha * logits_image_all
    #         logits_all = logits_all[:dataset_size]
    #         labels_all = labels_all[:dataset_size]


    #         topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))
    #         per_cat_correct = torch.zeros(40).to(self.config.device)
    #         per_cat_count = torch.zeros(40).to(self.config.device)
    #         for i in range(40):
    #             idx = (labels_all == i)
    #             if idx.sum() > 0:
    #                 per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
    #                 per_cat_count[i] = idx.sum()

    #         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    #         per_cat_acc = per_cat_correct / per_cat_count
    #         # for i in range(40):
    #         #    print(idx2category[i], per_cat_acc[i])

    #         if overall_acc > self.best_modelnet40_overall_acc:
    #             self.best_modelnet40_overall_acc = overall_acc
    #             # self.save_model('best_modelnet40_overall')
    #         if per_cat_acc.mean() > self.best_modelnet40_class_acc:
    #             self.best_modelnet40_class_acc = per_cat_acc.mean()
    #             # self.save_model('best_modelnet40_class')

    #         logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
    #                                                                                          self.best_modelnet40_overall_acc,
    #                                                                                          per_cat_acc.mean(),
    #                                                                                          self.best_modelnet40_class_acc))
    #         logging.info(
    #             'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
    #                                                                                 topk_acc[1].item(),
    #                                                                                 topk_acc[2].item()))

    #         # wandb.log({"test/epoch": self.epoch,
    #         #            "test/step": self.step,
    #         #            "test/ModelNet40_overall_acc": overall_acc,
    #         #            "test/ModelNet40_class_acc": per_cat_acc.mean(),
    #         #            "test/top3_acc": topk_acc[1],
    #         #            "test/top5_acc": topk_acc[2], })                                topk_acc[2].item()))

    #         torch.save({
    #                     "logits": logits_all,
    #                     "labels": labels_all,
    #                     "overall_acc": overall_acc,
    #                     "class_acc": per_cat_acc.mean()
    #                      },
    #                     os.path.join(self.config.ckpt_dir,f"modelnet40_epoch_{self.epoch}.pth")
    #                 )


    # def test_objaverse_lvis(self):
    #     self.model.eval()
    #     self.mrl_image_proj.eval() if self.config.training.use_mrl_image_proj else None
    #     self.mrl_text_proj.eval() if self.config.training.use_mrl_text_proj else None
    #     if self.config.training.use_text_proj:
    #         self.text_proj.eval()
    #     clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).to(self.config.device)
    #     if self.config.training.use_text_proj:
    #         clip_text_feat = self.text_proj(clip_text_feat)
    #     per_cat_correct = torch.zeros(1156).to(self.config.device)
    #     per_cat_count = torch.zeros(1156).to(self.config.device)
    #     category2idx = self.objaverse_lvis_loader.dataset.category2idx
    #     idx2category = {v: k for k, v in category2idx.items()}

    #     logits_all = []
    #     labels_all = []
    #     with torch.no_grad():
    #         for data in self.objaverse_lvis_loader:
    #             if not self.config.model.get("use_dense", False):
    #                 pred_feat = self.model(data['xyz'], data['features'], \
    #                                        device=self.config.device, \
    #                                        quantization_size=self.config.model.voxel_size)
    #             else:
    #                 pred_feat = self.model(data['xyz_dense'], data['features_dense'].to(self.config.device))

    #             labels = data['category'].to(self.config.device)

    #             current_logits_image = None
    #             current_logits_text = None

    #             if self.config.training.use_mrl_image_proj:
    #                 pred_feat_img = F.normalize(self.mrl_image_proj(pred_feat), dim=1)
    #                 current_logits_image = pred_feat_img @ F.normalize(clip_text_feat, dim=1).T


    #             if self.config.training.use_mrl_text_proj:
    #                 pred_feat_text = F.normalize(self.mrl_text_proj(clip_text_feat), dim=1)
    #                 current_logits_text = F.normalize(pred_feat, dim=1) @ pred_feat_text.T
              
    #             if current_logits_image is not None and current_logits_text is not None:
    #                 # final_logits = logits_text_all + self.alpha * logits_image_all
    #                 final_logits = (self.alpha * current_logits_image) + ((1 - self.alpha) * current_logits_text)
    #             elif current_logits_image is not None:
    #                 final_logits = current_logits_image
    #             elif current_logits_text is not None:
    #                 final_logits = current_logits_text
    #             else:
    #                 final_logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T

    #             logits_all.append(final_logits.detach())
    #             labels_all.append(labels)
    #     logits_all, labels_all = merge_results_dist(logits_all, labels_all)

    #     if self.rank == 0:
    #         dataset_size = len(self.objaverse_lvis_loader.dataset)
    #         logits_all = logits_all[:dataset_size]
    #         labels_all = labels_all[:dataset_size]
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
    #             # self.save_model('best_lvis')

    #         logging.info('Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
    #         logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
    #                                                                                             topk_acc[1].item(),
    #                                                                                             topk_acc[2].item()))
    #         # wandb.log({"test_lvis/epoch": self.epoch,
    #         #            "test_lvis/step": self.step,
    #         #            "test_lvis/overall_acc": overall_acc,
    #         #            "test_lvis/class_acc": per_cat_acc.mean(),
    #         #            "test_lvis/top3_acc": topk_acc[1],
    #         #            "test_lvis/top5_acc": topk_acc[2], })

    #         torch.save({
    #                     "logits": logits_all,
    #                     "labels": labels_all,
    #                     "overall_acc": overall_acc,
    #                     "class_acc": per_cat_acc.mean()
    #                      },
    #                     os.path.join(self.config.ckpt_dir,f"objaverse_lvis_epoch_{self.epoch}.pth")
    #                 )



    # def test_scanobjectnn(self):
    #     self.model.eval()
    #     self.mrl_image_proj.eval() if self.config.training.use_mrl_image_proj else None
    #     self.mrl_text_proj.eval() if self.config.training.use_mrl_text_proj else None
    #     if self.config.training.use_text_proj:
    #         self.text_proj.eval()
    #     clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
    #     if self.config.training.use_text_proj:
    #         clip_text_feat = self.text_proj(clip_text_feat)
    #     per_cat_correct = torch.zeros(15).to(self.config.device)
    #     per_cat_count = torch.zeros(15).to(self.config.device)
    #     category2idx = self.scanobjectnn_loader.dataset.category2idx
    #     idx2category = {v: k for k, v in category2idx.items()}

    #     logits_all = []
    #     labels_all = []
    #     with torch.no_grad():
    #         for data in self.scanobjectnn_loader:
    #             if not self.config.model.get("use_dense", False):
    #                 pred_feat = self.model(data['xyz'], data['features'], \
    #                                        device=self.config.device, \
    #                                        quantization_size=self.config.model.voxel_size)
    #             else:
    #                 pred_feat = self.model(data['xyz_dense'], data['features_dense'])

    #             labels = data['category'].to(self.config.device)

    #             current_logits_image = None
    #             current_logits_text = None

    #             if self.config.training.use_mrl_image_proj:
    #                 pred_feat_img = F.normalize(self.mrl_image_proj(pred_feat), dim=1)
    #                 current_logits_image = pred_feat_img @ F.normalize(clip_text_feat, dim=1).T


    #             if self.config.training.use_mrl_text_proj:
    #                 pred_feat_text = F.normalize(self.mrl_text_proj(clip_text_feat), dim=1)
    #                 current_logits_text = F.normalize(pred_feat, dim=1) @ pred_feat_text.T
              
    #             if current_logits_image is not None and current_logits_text is not None:
    #                 # final_logits = logits_text_all + self.alpha * logits_image_all
    #                 final_logits = (self.alpha * current_logits_image) + ((1 - self.alpha) * current_logits_text)
    #             elif current_logits_image is not None:
    #                 final_logits = current_logits_image
    #             elif current_logits_text is not None:
    #                 final_logits = current_logits_text
    #             else:
    #                 final_logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T

    #             logits_all.append(final_logits.detach())
    #             labels_all.append(labels)

    #     logits_all, labels_all = merge_results_dist(logits_all, labels_all)

    #     if self.rank == 0:
    #         dataset_size = len(self.scanobjectnn_loader.dataset)
    #         logits_all = logits_all[:dataset_size]
    #         labels_all = labels_all[:dataset_size]
    #         topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

    #         # calculate per class accuracy
    #         for i in range(15):
    #             idx = (labels_all == i)
    #             if idx.sum() > 0:
    #                 per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
    #                 per_cat_count[i] = idx.sum()

    #         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    #         per_cat_acc = per_cat_correct / per_cat_count

    #         logging.info('Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
    #         logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
    #                                                                                            topk_acc[1].item(),
    #                                                                                            topk_acc[2].item()))
    #         # wandb.log({"test_scanobjectnn/epoch": self.epoch,
    #         #            "test_scanobjectnn/step": self.step,
    #         #            "test_scanobjectnn/overall_acc": overall_acc,
    #         #            "test_scanobjectnn/class_acc": per_cat_acc.mean(),
    #         #            "test_scanobjectnn/top3_acc": topk_acc[1],
    #         #            "test_scanobjectnn/top5_acc": topk_acc[2], })
    #         torch.save({
    #                     "logits": logits_all,
    #                     "labels": labels_all,
    #                     "overall_acc": overall_acc,
    #                     "class_acc": per_cat_acc.mean()
    #                      },
    #                     os.path.join(self.config.ckpt_dir,f"scanobjectnn_epoch_{self.epoch}.pth")
    #                 )   


    # def test_scannet(self, scannet_loader):
    #     self.model.eval()
    #     self.mrl_image_proj.eval() if self.config.training.use_mrl_image_proj else None
    #     self.mrl_text_proj.eval() if self.config.training.use_mrl_text_proj else None
    #     if self.config.training.use_text_proj:
    #         self.text_proj.eval()
    #     clip_text_feat = torch.from_numpy(scannet_loader.dataset.clip_cat_feat).to(self.config.device)
    #     if self.config.training.use_text_proj:
    #         clip_text_feat = self.text_proj(clip_text_feat)
    #     per_cat_correct = torch.zeros(19).to(self.config.device)
    #     per_cat_count = torch.zeros(19).to(self.config.device)
    #     category2idx = scannet_loader.dataset.category2idx
    #     idx2category = {v: k for k, v in category2idx.items()}

    #     logits_all = []
    #     labels_all = []
    #     with torch.no_grad():
    #         for data in tqdm(scannet_loader):
    #             if not self.config.model.get("use_dense", False):
    #                 pred_feat = self.model(data['xyz'], data['features'], \
    #                                        device=self.config.device, \
    #                                        quantization_size=self.config.model.voxel_size)
    #             else:
    #                 pred_feat = self.model(data['xyz_dense'], data['features_dense'])

    #             labels = data['category'].to(self.config.device)

    #             current_logits_image = None
    #             current_logits_text = None

    #             if self.config.training.use_mrl_image_proj:
    #                 pred_feat_img = F.normalize(self.mrl_image_proj(pred_feat), dim=1)
    #                 current_logits_image = pred_feat_img @ F.normalize(clip_text_feat, dim=1).T


    #             if self.config.training.use_mrl_text_proj:
    #                 pred_feat_text = F.normalize(self.mrl_text_proj(clip_text_feat), dim=1)
    #                 current_logits_text = F.normalize(pred_feat, dim=1) @ pred_feat_text.T
              
    #             if current_logits_image is not None and current_logits_text is not None:
    #                 # final_logits = logits_text_all + self.alpha * logits_image_all
    #                 final_logits = (self.alpha * current_logits_image) + ((1 - self.alpha) * current_logits_text)
    #             elif current_logits_image is not None:
    #                 final_logits = current_logits_image
    #             elif current_logits_text is not None:
    #                 final_logits = current_logits_text
    #             else:
    #                 final_logits = F.normalize(pred_feat, dim=1) @ F.normalize(clip_text_feat, dim=1).T

    #             logits_all.append(final_logits.detach())
    #             labels_all.append(labels)

    #     logits_all, labels_all = merge_results_dist(logits_all, labels_all)

    #     if self.rank == 0:
    #         dataset_size = len(self.scannet_loader.dataset)
    #         logits_all = logits_all[:dataset_size]
    #         labels_all = labels_all[:dataset_size]

    #         topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

    #         # calculate per class accuracy
    #         for i in range(19):
    #             idx = (logits_all == i)
    #             if idx.sum() > 0:
    #                 per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
    #                 per_cat_count[i] = idx.sum()

    #         overall_acc = per_cat_correct.sum() / per_cat_count.sum()
    #         per_cat_acc = per_cat_correct / per_cat_count


    #         logging.info('Test Scannet: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
    #         logging.info('Test Scannet: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
    #                                                                                        topk_acc[1].item(),
    #                                                                                            topk_acc[2].item()))

    #         torch.save({
    #                     "logits": logits_all,  
    #                     "labels": labels_all,
    #                     "overall_acc": overall_acc,
    #                     "class_acc": per_cat_acc.mean()
    #                         },
    #                     os.path.join(self.config.ckpt_dir,f"scannet_epoch_{self.epoch}.pth"))
            