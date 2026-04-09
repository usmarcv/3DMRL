import logging
import os
import random
import shutil
import sys
from datetime import datetime

import MinkowskiEngine as ME
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import data
import models
from models.LogitScaleNetwork import LogitScaleNetwork
from models.clip_adapter import NewCLIP
from trainers.mlp import MLP, MLP_ME
from trainers.trainer import Trainer
from trainers.tamm_trainer import TAMM_Trainer
from trainers.tamm_trainer_mrl import TAMM_Trainer_MRL
from trainers.clip_adapter_trainer import CLIP_Adapter_Trainer
from trainers.mrl_alignment import Trainer_MRL

# ── NOVO: importa o trainer e as heads MRL ──────────────────────────────────
from trainers.mrl_trainer_adapters import MRL_Trainer, MRLProjectionHeads
# ────────────────────────────────────────────────────────────────────────────

from utils.logger import setup_logging
from utils.misc import load_config, dump_config
from utils.scheduler import cosine_lr, const_lr
from param import parse_args


def main(cli_args, extras):

    # ── Init DDP ─────────────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl")

    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    device = torch.device("cuda", local_rank)

    # ── Config ───────────────────────────────────────────────────────────────
    config = load_config(cli_args.config, cli_args=vars(cli_args), extra_args=extras)
    if config.autoresume:
        config.trial_name = config.get('trial_name') + "@autoresume"
    else:
        config.trial_name = config.get('trial_name') + datetime.now().strftime('@%Y%m%d-%H%M%S')
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')

    config.ngpu   = world_size
    config.device = f"cuda:{local_rank}"

    if config.fix_seed:
        seed = config.seed
        torch.manual_seed(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)

    if rank == 0:
        os.makedirs(os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        if os.path.exists(config.code_dir):
            shutil.rmtree(config.code_dir)
        config.log_path  = config.get('log_path') or os.path.join(config.exp_dir, config.trial_name, 'log.txt')
        config.log_level = logging.DEBUG if config.debug else logging.INFO
        setup_logging(config.log_path, config.log_level)
        dump_config(os.path.join(config.exp_dir, config.trial_name, 'config.yaml'), config)
        logging.info("Using {} GPU(s).".format(config.ngpu))

    # ── Treino ───────────────────────────────────────────────────────────────
    if config.train:

        # ── Backbone 3D ──────────────────────────────────────────────────────
        model = models.make(config).to(device)
        if config.model.name.startswith('Mink'):
            model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
            if rank == 0:
                logging.info("Using MinkowskiSyncBatchNorm")
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            if rank == 0:
                logging.info("Using SyncBatchNorm")

        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            logging.info("Network: {}, params: {}".format(config.model.name, total_params))

        # ── Módulos auxiliares ───────────────────────────────────────────────
        logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(device)
        image_proj  = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device)
        text_proj   = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device)

        # ── NOVO: MRLProjectionHeads substitui mrl_nested_proj ──────────────
        #
        #   config.mrl.dims    : [8, 16, 32, 64, 128, 256, 512]
        #   config.clip_embed_dim : 1280  (dimensão do espaço CLIP professor)
        #
        #   Cada head_dim : Linear(dim → 1280, bias=False)
        #   Entrada       : z[:, :dim]  — prefixo do CLS token do PointBERT
        #   Saída         : ℝ^1280 normalizado — comparado diretamente com CLIP
        #
        mrl_heads = MRLProjectionHeads(
            mrl_dims=config.mrl.dims,
            clip_dim=config.clip_embed_dim,   # 1280
        ).to(device)

        if rank == 0:
            mrl_params = sum(p.numel() for p in mrl_heads.parameters())
            logging.info("MRLProjectionHeads: {} granularidades, {} parâmetros".format(
                len(config.mrl.dims), mrl_params
            ))
        # ────────────────────────────────────────────────────────────────────

        # ── MLPs de alinhamento (apenas para trainer mrl_alignment) ─────────
        mlp_type = config.training.get('mlp_type')
        image_alignment_adapter = None
        text_alignment_adapter  = None

        if mlp_type == "mlp":
            image_alignment_adapter = MLP(
                in_features=config.training.image_branch_in_dim,
                hidden_features=config.training.image_branch_hidden,
                out_features=config.training.image_branch_out_dim,
                drop=config.training.image_branch_dropout,
                activate=config.training.activate,
            ).to(device)
            text_alignment_adapter = MLP(
                in_features=config.training.text_branch_in_dim,
                hidden_features=config.training.text_branch_hidden,
                out_features=config.training.text_branch_out_dim,
                drop=config.training.text_branch_dropout,
                activate=config.training.activate,
            ).to(device)
        elif mlp_type == "mlp_me":
            image_alignment_adapter = MLP_ME(
                in_features=config.training.image_branch_in_dim,
                hidden_features=config.training.image_branch_hidden,
                out_features=config.training.image_branch_out_dim,
                drop=config.training.image_branch_dropout,
            ).to(device)
            text_alignment_adapter = MLP_ME(
                in_features=config.training.text_branch_in_dim,
                hidden_features=config.training.text_branch_hidden,
                out_features=config.training.text_branch_out_dim,
                drop=config.training.image_branch_dropout,
            ).to(device)

        # ── DDP wrapping ─────────────────────────────────────────────────────
        logit_scale = DDP(logit_scale, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        image_proj  = DDP(image_proj,  device_ids=[rank], output_device=rank, find_unused_parameters=False)
        text_proj   = DDP(text_proj,   device_ids=[rank], output_device=rank, find_unused_parameters=False)
        mrl_heads   = DDP(mrl_heads,   device_ids=[rank], output_device=rank, find_unused_parameters=False)

        if mlp_type:
            image_alignment_adapter = DDP(image_alignment_adapter, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            text_alignment_adapter  = DDP(text_alignment_adapter,  device_ids=[rank], output_device=rank, find_unused_parameters=False)

        # ── Data loaders ─────────────────────────────────────────────────────
        train_loader          = data.make(config, 'train', rank, world_size)
        modelnet40_loader     = data.make_modelnet40test(config)
        objaverse_lvis_loader = data.make_objaverse_lvis(config)
        scanobjectnn_loader   = data.make_scanobjectnntest(config)

        if rank == 0 and train_loader is not None:
            logging.info("Train iterations: {}".format(len(train_loader)))

        # ── Optimizer ────────────────────────────────────────────────────────
        #
        #   As mrl_heads recebem lr * 2 porque começam do zero
        #   enquanto o PointBERT já tem pretrain.
        #
        if config.trainer == "mrl_trainer":
            params = [
                {"params": model.parameters()},
                {"params": image_proj.parameters()},
                {"params": text_proj.parameters()},
                {"params": mrl_heads.parameters(), "lr": config.training.lr * world_size * 2},
                {"params": logit_scale.parameters()},
            ]
        elif config.trainer == "mrl_alignment":
            assert image_alignment_adapter is not None
            assert text_alignment_adapter  is not None
            params = [
                {"params": model.parameters()},
                {"params": image_proj.parameters()},
                {"params": text_proj.parameters()},
                {"params": mrl_heads.parameters(), "lr": config.training.lr * world_size * 2},
                {"params": image_alignment_adapter.parameters()},
                {"params": text_alignment_adapter.parameters()},
                {"params": logit_scale.parameters()},
            ]
        else:
            raise ValueError("Trainer '{}' não reconhecido.".format(config.trainer))

        # ── Scheduler ────────────────────────────────────────────────────────
        #   NÃO MEXER NESSA DESGRAÇA ******************************************
        if config.training.scheduler == "cosine":
            lr = config.training.lr * world_size
            optimizer    = torch.optim.AdamW(
                params, lr=lr,
                betas=(config.training.beta1, config.training.beta2),
                eps=config.training.eps,
            )
            warmup_steps = config.training.warmup_epoch * len(train_loader)
            total_steps  = config.training.max_epoch   * len(train_loader)
            scheduler    = cosine_lr(optimizer, lr, warmup_steps, total_steps)
        elif config.training.scheduler == "const":
            lr           = config.training.lr * world_size
            optimizer    = torch.optim.AdamW(
                params, lr=lr,
                betas=(config.training.beta1, config.training.beta2),
                eps=config.training.eps,
            )
            warmup_steps = config.training.warmup_epoch * len(train_loader)
            total_steps  = config.training.max_epoch   * len(train_loader)
            scheduler    = const_lr(optimizer, lr, warmup_steps, total_steps)
        else:
            lr_decay_step = config.training.lr_decay * len(train_loader)
            optimizer     = torch.optim.AdamW(params, lr=config.training.lr)
            scheduler     = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=lr_decay_step, gamma=config.training.lr_decay_rate
            )
        # ********************************************************************

        # ── Instancia o Trainer ──────────────────────────────────────────────
        if config.trainer == "mrl_trainer":
            trainer = MRL_Trainer(
                rank=rank,
                config=config,
                model=model,
                logit_scale=logit_scale,
                image_proj=image_proj,
                text_proj=text_proj,
                mrl_heads=mrl_heads,          # <── substituiu mrl_nested_proj
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                modelnet40_loader=modelnet40_loader,
                objaverse_lvis_loader=objaverse_lvis_loader,
                scanobjectnn_loader=scanobjectnn_loader,
            )

        elif config.trainer == "mrl_alignment":
            trainer = Trainer_MRL(
                rank=rank,
                config=config,
                model=model,
                logit_scale=logit_scale,
                image_proj=image_proj,
                text_proj=text_proj,
                mrl_heads=mrl_heads,          # <── idem
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                image_alignment_adapter=image_alignment_adapter,
                text_alignment_adapter=text_alignment_adapter,
                modelnet40_loader=modelnet40_loader,
                objaverse_lvis_loader=objaverse_lvis_loader,
                scanobjectnn_loader=scanobjectnn_loader,
            )

        # ── Resume ───────────────────────────────────────────────────────────
        if config.resume is not None:
            trainer.load_from_checkpoint(config.resume)
        elif config.autoresume:
            latest = os.path.join(config.ckpt_dir, 'latest.pt')
            if os.path.exists(latest):
                trainer.load_from_checkpoint(latest)

        trainer.train()

    # ── End DDP ──────────────────────────────────────────────────────────────
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    cli_args, extras = parse_args(sys.argv[1:])
    main(cli_args, extras)

# import logging
# import os
# import random
# import shutil
# import sys
# from datetime import datetime

# import MinkowskiEngine as ME
# import numpy as np
# import torch
# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

# import data
# import models
# from models.LogitScaleNetwork import LogitScaleNetwork
# from models.clip_adapter import NewCLIP
# from trainers.mlp import MLP, MLP_ME
# from trainers.trainer import Trainer
# from trainers.tamm_trainer import TAMM_Trainer
# from trainers.tamm_trainer_mrl import TAMM_Trainer_MRL
# from trainers.clip_adapter_trainer import CLIP_Adapter_Trainer

# # from trainers.alignment_mrl import Alignment_MRL

# from trainers.mrl_trainer_adapters import MRL_Trainer

# from trainers.mrl_alignment import Trainer_MRL #pc alignment

# from utils.logger import setup_logging
# from utils.misc import load_config, dump_config
# from utils.scheduler import cosine_lr, const_lr
# from param import parse_args


# def main(cli_args, extras):

#     #Init DDP
#     dist.init_process_group(backend="nccl")

#     rank = dist.get_rank()
#     world_size = dist.get_world_size() # number of processes = number of gpus
#     local_rank = int(os.environ["LOCAL_RANK"])

#     torch.cuda.set_device(local_rank)
    
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

#     device = torch.device("cuda", local_rank)


#     config = load_config(cli_args.config, cli_args=vars(cli_args), extra_args=extras)
#     if config.autoresume:
#         config.trial_name = config.get('trial_name') + "@autoresume"
#     else:
#         config.trial_name = config.get('trial_name') + datetime.now().strftime('@%Y%m%d-%H%M%S')
#     config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
#     config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')

#     config.ngpu = world_size
#     config.device = f"cuda:{local_rank}"

#     # fix the seed
#     if config.fix_seed:
#         seed = config.seed
#         torch.manual_seed(seed + rank)
#         np.random.seed(seed + rank)
#         random.seed(seed + rank)

#     if rank == 0:
#         os.makedirs(os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume)
#         os.makedirs(config.ckpt_dir, exist_ok=True)
#         if os.path.exists(config.code_dir):
#             shutil.rmtree(config.code_dir)

#     # config.device = 'cuda:{0}'.format(rank)
    
#     if rank == 0:
#         config.log_path = config.get('log_path') or os.path.join(config.exp_dir, config.trial_name, 'log.txt')
#         config.log_level = logging.DEBUG if config.debug else logging.INFO
#         setup_logging(config.log_path, config.log_level)
#         dump_config(os.path.join(config.exp_dir, config.trial_name, 'config.yaml'), config)
#         logging.info("Using {} GPU(s).".format(config.ngpu))

#     if config.train:
#         model = models.make(config).to(device)    
#         if config.model.name.startswith('Mink'):
#             model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)  # minkowski only
#             if rank == 0:
#                 logging.info("Using MinkowskiSyncBatchNorm")
#         else:
#             model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
#             if rank == 0:
#                 logging.info("Using SyncBatchNorm")

#         model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        
#         if rank == 0:
#             total_params = sum(p.numel() for p in model.parameters())
#             logging.info("Network: {}, Number of parameters: {}".format(config.model.name, total_params))

#         logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(device)
#         image_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device)
#         text_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device)
#         mrl_image_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device)
#         mrl_text_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device)
#         mrl_nested_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel, bias=False).to(device)

#         mlp_type = config.training.get('mlp_type')
#         if mlp_type == "mlp":
#             image_alignment_adapter = MLP(in_features=config.training.image_branch_in_dim,
#                                  hidden_features=config.training.image_branch_hidden,
#                                  out_features=config.training.image_branch_out_dim,
#                                  drop=config.training.image_branch_dropout,
#                                  activate=config.training.activate).to(device)
#             text_alignment_adapter = MLP(in_features=config.training.text_branch_in_dim,
#                                 hidden_features=config.training.text_branch_hidden,
#                                 out_features=config.training.text_branch_out_dim,
#                                 drop=config.training.text_branch_dropout,
#                                 activate=config.training.activate).to(device)
#         elif mlp_type == "mlp_me":
#             image_alignment_adapter = MLP_ME(in_features=config.training.image_branch_in_dim,
#                                   hidden_features=config.training.image_branch_hidden,
#                                   out_features=config.training.image_branch_out_dim, drop=config.training.image_branch_dropout).to(device)
#             text_alignment_adapter = MLP_ME(in_features=config.training.text_branch_in_dim,
#                                  hidden_features=config.training.text_branch_hidden,
#                                  out_features=config.training.text_branch_out_dim, drop=config.training.image_branch_dropout).to(device)

#         logit_scale = DDP(logit_scale, device_ids=[rank], output_device=rank, find_unused_parameters=False)
#         image_proj = DDP(image_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
#         text_proj = DDP(text_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
#         mrl_image_proj = DDP(mrl_image_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
#         mrl_text_proj = DDP(mrl_text_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
#         mrl_nested_proj = DDP(mrl_nested_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)

#         train_loader = data.make(config, 'train', rank, world_size)

#         if mlp_type:
#             image_alignment_adapter = DDP(image_alignment_adapter, device_ids=[rank], output_device=rank, find_unused_parameters=False)
#             text_alignment_adapter = DDP(text_alignment_adapter, device_ids=[rank], output_device=rank, find_unused_parameters=False)


#         modelnet40_loader = data.make_modelnet40test(config)
#         objaverse_lvis_loader = data.make_objaverse_lvis(config)
#         scanobjectnn_loader = data.make_scanobjectnntest(config)

  

#         if rank == 0 and train_loader is not None:
#                 logging.info("Train iterations: {}".format(len(train_loader)))

#         # testando projetar direto do espaco
#         if config.trainer == "mrl_trainer":
#             params = list(model.parameters()) + list(image_proj.parameters()) + list(text_proj.parameters()) + \
#                      list(mrl_image_proj.parameters()) + list(mrl_text_proj.parameters()) + \
#                      list(mrl_nested_proj.parameters()) + \
#                      list(logit_scale.parameters())
#         elif config.trainer == "mrl_alignment":
#             params = list(model.parameters()) + list(image_proj.parameters()) + list(text_proj.parameters()) + \
#                      list(mrl_image_proj.parameters()) + list(mrl_text_proj.parameters()) + \
#                      list(mrl_nested_proj.parameters()) + \
#                      list(image_alignment_adapter.parameters()) + list(text_alignment_adapter.parameters()) + \
#                      list(logit_scale.parameters()) 


#         #NÃO MEXER MAIS NESSSA DISSSSSSSGRAAAAAAÇA *****************************************************
#         # lr = config.training.lr * world_size
#         # lr = config.training.lr


#         if config.training.scheduler == "cosine":
#             lr = config.training.lr * world_size
#             optimizer = torch.optim.AdamW(
#                         params,
#                         lr=lr,
#                         betas=(config.training.beta1, config.training.beta2),
#                         eps=config.training.eps
#                     )
#             warmup_steps = config.training.warmup_epoch * len(train_loader)
#             total_steps = config.training.max_epoch * len(train_loader)
#             scheduler = cosine_lr(optimizer, lr, warmup_steps, total_steps)
#         elif config.training.scheduler == "const":
#             scheduler = const_lr(optimizer, config.training.lr, warmup_steps, total_steps)
#         else:
#             lr_decay_step = config.training.lr_decay * len(train_loader)
#             scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step,
#                                                         gamma=config.training.lr_decay_rate)
#         # ************************************************************************************************
       
#         #Choose your type of trainer here   
#         try:
#             if config.trainer == "mrl_trainer":
#                 trainer = MRL_Trainer(rank, config, model, logit_scale, \
#                                         image_proj, text_proj, mrl_image_proj, mrl_text_proj, mrl_nested_proj,
#                                         optimizer, scheduler, train_loader, \
#                                         modelnet40_loader, objaverse_lvis_loader, scanobjectnn_loader)

#             elif config.trainer == "mrl_alignment":
#                 assert image_alignment_adapter is not None
#                 assert text_alignment_adapter is not None   
#                 trainer = Trainer_MRL(rank=rank,
#                             config=config,
#                             model=model,
#                             logit_scale=logit_scale,
#                             image_proj=image_proj,
#                             text_proj=text_proj,
#                             mrl_image_proj=mrl_image_proj,
#                             mrl_text_proj=mrl_text_proj,
#                             mrl_nested_proj=mrl_nested_proj,
#                             optimizer=optimizer,
#                             scheduler=scheduler,
#                             train_loader=train_loader,
#                             image_alignment_adapter=image_alignment_adapter,
#                             text_alignment_adapter=text_alignment_adapter,
#                             modelnet40_loader=modelnet40_loader,
#                             objaverse_lvis_loader=objaverse_lvis_loader,
#                             scanobjectnn_loader=scanobjectnn_loader
#                         )
#             else:
#                 raise ValueError("Trainer {} not recognized.".format(config.trainer))


#         except ValueError as e:
#             raise ValueError(e)


#         if config.resume is not None:
#             trainer.load_from_checkpoint(config.resume)
#         elif config.autoresume:
#             if os.path.exists(os.path.join(config.ckpt_dir, '{}.pt'.format('latest'))):
#                 trainer.load_from_checkpoint(os.path.join(config.ckpt_dir, '{}.pt'.format('latest')))

#         trainer.train()

#     #End DDP
#     dist.barrier()
#     dist.destroy_process_group()


# if __name__ == '__main__':
#     cli_args, extras = parse_args(sys.argv[1:])
#     main(cli_args, extras)