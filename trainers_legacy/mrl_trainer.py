import logging
import os
import numpy as np
import torch
import torch.distributed.nn
import torch.nn.functional as F
import wandb
import torch.distributed as dist
from tqdm import tqdm
from trainers.trainer_utils import merge_two_branch_results_dist, merge_results_dist
import math
from collections import OrderedDict, defaultdict


from trainers.MRL import MRL_Contrastive_Loss

class MRL_Trainer(object):
    def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, mrl_image_proj, mrl_text_proj,
                    optimizer, scheduler, train_loader, \
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
        # self.mrl_nested_proj = mrl_nested_proj
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.epoch = 0
        self.step = 0
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0
        self.mrl_criterion = MRL_Contrastive_Loss(config=self.config, nesting_list=self.config.nesting_list)
        self.config.ngpu = dist.get_world_size()


    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.logit_scale.load_state_dict(checkpoint['logit_scale'])  # module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.use_openclip_optimizer_scheduler == False:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))



    def train_one_epoch(self):  
        self.model.train()
        # self.mrl_nested_proj.train()
        if self.config.training.use_text_proj: #True
            self.text_proj.train() 
        if self.config.training.use_image_proj: 
            self.image_proj.train()

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
                pred_feat = self.model(
                    data['xyz'], data['features'], \
                    device=self.config.device, \
                    quantization_size=self.config.model.voxel_size
                )
            else:
                pred_feat = self.model(
                    data['xyz_dense'], 
                    data['features_dense']
                )

            logit_scale = self.logit_scale(None)
            
            #collect the image and text features
            idx = data['has_text_idx']
            text_feat = torch.vstack(data['text_feat']).to(self.config.device)
            img_feat = torch.vstack(data['img_feat']).to(self.config.device)

            if self.config.training.use_mask:
                # usa só a primeira imagem para calcular similaridade img-text
                first_img = img_feat[:, :self.config.clip_embed_dim]
                img_text_sim = (
                    F.normalize(first_img, dim=-1) @
                    F.normalize(text_feat, dim=-1).T
                )
                mask = (
                    torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim
                    > self.config.training.mask_threshold
                )
                mask = torch.logical_or(mask, mask_other).detach()
            else:
                mask = None


            # **************************** Multimodal Learning here *******************************************************

            image_acc_mean = 0.0
            text_acc_mean = 0.0
            #feat number of images
            if self.config.dataset.num_imgs > 0:
                img_acc_accum = 0.0
                for i in range(self.config.dataset.num_imgs):
                    single_image_feat = img_feat[ : , i * self.config.clip_embed_dim : (i + 1) * self.config.clip_embed_dim].to(self.config.device)
                    # single_image_feat = img_feat.to(self.config.device)

                    if self.config.training.use_image_proj:
                        single_image_feat = self.image_proj(single_image_feat)

                    if self.config.training.use_mrl_image_proj:
                        single_image_feat = self.mrl_image_proj(single_image_feat)

                    img_contras_loss, img_contras_acc, \
                    img_loss_dims, img_acc_dims =  self.mrl_criterion(
                                                        pred_feat, single_image_feat,
                                                        logit_scale=logit_scale,
                                                        mask=mask)

                    loss += img_contras_loss * self.config.training.lambda_img_contras
                    # img_contras_acc_list.append(img_contras_acc.item())
                    img_acc_accum += img_contras_acc

                    for d, val in img_loss_dims.items():
                        epoch_img_loss_dim[d].append(val)
                    for d, val in img_acc_dims.items():
                        epoch_img_acc_dim[d].append(val)
                
                img_acc_mean = img_acc_accum / self.config.dataset.num_imgs

            #feat number of texts
            if len(idx) > 0:
                if self.config.training.use_text_proj:
                    text_feat = self.text_proj(text_feat)
                if self.config.training.use_mrl_text_proj:
                    text_feat = self.mrl_text_proj(text_feat)

                text_contras_loss, text_contras_acc, \
                txt_loss_dims, txt_acc_dims = self.mrl_criterion(
                                                pred_feat[idx], text_feat,
                                                logit_scale=logit_scale, 
                                                mask=mask)

                loss += text_contras_loss * self.config.training.lambda_text_contras
                text_acc_mean = text_contras_acc

                for d, val in txt_loss_dims.items():
                    epoch_txt_loss_dim[d].append(val)
                for d, val in txt_acc_dims.items():
                    epoch_txt_acc_dim[d].append(val)
            
            # *******************************************************************************************

            if self.config.dataset.num_imgs > 0:
                img_contras_acc_list.append(img_acc_mean)
            if len(idx) > 0:
                text_contras_acc_list.append(text_acc_mean)

            loss.backward()
            self.optimizer.step()

            if self.config.training.scheduler == "cosine" or self.config.training.scheduler == "const":
                self.scheduler(self.step)
            else:
                self.scheduler.step()
        
        if self.rank == 0:
            logging.info('Train avg: text_constrat_acc: {0} image_contrast_acc: {1}' \
                         .format((np.mean(text_contras_acc_list)) if len(text_contras_acc_list) > 0 else 0,
                                 (np.mean(img_contras_acc_list)) if len(img_contras_acc_list) > 0 else 0))

            header = f"{'Dim':<6} | {'Img Loss':<10} | {'Img Acc':<10} | {'Txt Loss':<10} | {'Txt Acc':<10}"
            logging.info("-" * len(header))
            logging.info(header)
            logging.info("-" * len(header))
            
            all_dims = sorted(set(epoch_img_loss_dim.keys()) | set(epoch_txt_loss_dim.keys()), key=lambda x: int(x.replace('loss_', '').replace('d', '')))
            
            for d in all_dims:
                # Ajustando a chave para buscar corretamente no dicionário de loss e acc
                dim_number = d.replace('loss_', '').replace('d', '')
                
                i_loss = np.mean(epoch_img_loss_dim.get(f"loss_{dim_number}d", [0]))
                i_acc  = np.mean(epoch_img_acc_dim.get(f"acc_{dim_number}d", [0]))
                t_loss = np.mean(epoch_txt_loss_dim.get(f"loss_{dim_number}d", [0]))
                t_acc  = np.mean(epoch_txt_acc_dim.get(f"acc_{dim_number}d", [0]))
                
                logging.info(f"{dim_number+'d':<6} | {i_loss:.4f}     | {i_acc:.4f}     | {t_loss:.4f}     | {t_acc:.4f}")
            
            logging.info("-" * len(header))

        # if self.rank == 0:
        #     logging.info('Train avg: text_constrat_acc: {0} image_contrast_acc: {1}' \
        #                  .format((np.mean(text_contras_acc_list)) if len(text_contras_acc_list) > 0 else 0,
        #                          (np.mean(img_contras_acc_list)) if len(img_contras_acc_list) > 0 else 0))

                    
        #     header = f"{'Dim':<6} | {'Img Loss':<10} | {'Img Acc':<10} | {'Txt Loss':<10} | {'Txt Acc':<10}"
        #     logging.info("-" * len(header))
        #     logging.info(header)
        #     logging.info("-" * len(header))
            
        #     all_dims = sorted(set(epoch_img_loss_dim.keys()) | set(epoch_txt_loss_dim.keys()))
            
        #     for d in all_dims:
        #         i_loss = np.mean(epoch_img_loss_dim.get(d, [0]))
        #         i_acc  = np.mean(epoch_img_acc_dim.get(d, [0]))
        #         t_loss = np.mean(epoch_txt_loss_dim.get(d, [0]))
        #         t_acc  = np.mean(epoch_txt_acc_dim.get(d, [0]))
                
        #         logging.info(f"{d:<6} | {i_loss:.4f}     | {i_acc:.4f}     | {t_loss:.4f}     | {t_acc:.4f}")
            
        #     logging.info("-" * len(header))


    def save_model(self, name):
        torch.save({
            "state_dict": self.model.state_dict(),
            "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
            "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
            "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
            "optimizer": self.optimizer.state_dict(),
            # "mrl_nested_proj": self.mrl_nested_proj.state_dict() if self.config.training.use_mrl_nested_proj else None,
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
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)

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
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
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