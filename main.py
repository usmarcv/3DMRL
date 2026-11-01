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



# our new era here
from trainers.MRL import MRL_Projection_Layer
from trainers.trainer_3dmrl import TrainerToMRL


from utils.logger import setup_logging
from utils.misc import load_config, dump_config
from utils.scheduler import cosine_lr, const_lr
from param import parse_args


def main(cli_args, extras):

    # Init DDP
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size() # number of processes = number of gpus
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda", local_rank)

    config = load_config(cli_args.config, cli_args=vars(cli_args), extra_args=extras)
    
    if config.autoresume:
        config.trial_name = config.get('trial_name') + "@autoresume"
    else:
        config.trial_name = config.get('trial_name') + datetime.now().strftime('@%Y%m%d-%H%M%S')
    
    config.ckpt_dir = config.get('ckpt_dir') or os.path.join(config.exp_dir, config.trial_name, 'ckpt')
    config.code_dir = config.get('code_dir') or os.path.join(config.exp_dir, config.trial_name, 'code')

    config.ngpu = world_size
    config.device = f"cuda:{local_rank}"

    # Fix the seed
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
    
    if rank == 0:
        config.log_path = config.get('log_path') or os.path.join(config.exp_dir, config.trial_name, 'log.txt')
        config.log_level = logging.DEBUG if config.debug else logging.INFO
        setup_logging(config.log_path, config.log_level)
        dump_config(os.path.join(config.exp_dir, config.trial_name, 'config.yaml'), config)
        logging.info("Using {} GPU(s).".format(config.ngpu))

    if config.train:
        # ==============================================================================
        # 1. SETUP DE DADOS COMUM E LOGIT SCALE
        # ==============================================================================
        train_loader = data.make(config, 'train', rank, world_size)
        
        # Loaders de teste
        modelnet40_loader = data.make_modelnet40test(config)
        objaverse_lvis_loader = data.make_objaverse_lvis(config)
        scanobjectnn_loader = data.make_scanobjectnntest(config)

        if rank == 0 and train_loader is not None:
            logging.info("Train iterations: {}".format(len(train_loader)))

        logit_scale = LogitScaleNetwork(config.training.logit_scale_init).to(device)
        logit_scale = DDP(logit_scale, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        # Variável para guardar os parâmetros que serão treinados
        params_to_optimize = []

        # ==============================================================================
        # 2. INSTANCIAÇÃO DOS MODELOS (BASEADO NO ESTÁGIO)
        # ==============================================================================
        

        if config.trainer == "3dmrl_trainer":
            if rank == 0:
                logging.info("--- Starting 3DMRL trainer... ---")

            model = models.make(config).to(device)    
            if config.model.name.startswith('Mink'):
                model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
                if rank == 0: logging.info("Using MinkowskiSyncBatchNorm")
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if rank == 0: logging.info("Using SyncBatchNorm")

            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            
            if rank == 0:
                total_params = sum(p.numel() for p in model.parameters())
                logging.info(f"Network: {config.model.name}, Parâmetros de Treino: {total_params}")
            
            image_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device)
            text_proj = torch.nn.Linear(config.model.out_channel, config.model.out_channel).to(device) 
            
            image_proj = DDP(image_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            text_proj = DDP(text_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)


            mrl_heads = MRL_Projection_Layer(nesting_list=config.mrl.nesting_dims,
                                            out_dim=config.mrl.out_dim,
                                            efficient=config.mrl.efficient
                                            ).to(device)

            mrl_heads = DDP(mrl_heads,   device_ids=[rank], output_device=rank, find_unused_parameters=False)


            params_to_optimize =  list(model.parameters()) + list(logit_scale.parameters()) + list(mrl_heads.parameters())

            if config.training.use_image_proj:
                params_to_optimize += list(image_proj.parameters())
            if config.training.use_text_proj:
                params_to_optimize += list(text_proj.parameters())

        else:
            raise ValueError(f"Trainer '{config.trainer}' não reconhecido. Use '3dmrl_trainer'.")


        # ==============================================================================
        # 3. OTIMIZADOR E SCHEDULER COMUM
        # ==============================================================================
        if config.training.scheduler == "cosine":
            # lr = config.training.lr * world_size
            lr = config.training.lr
            optimizer = torch.optim.AdamW(
                        params_to_optimize,
                        lr=lr,
                        betas=(config.training.beta1, config.training.beta2),
                        eps=config.training.eps
                    )
            warmup_steps = config.training.warmup_epoch * len(train_loader)
            total_steps = config.training.max_epoch * len(train_loader)
            scheduler = cosine_lr(optimizer, lr, warmup_steps, total_steps)
        elif config.training.scheduler == "const":
            optimizer = torch.optim.AdamW(params_to_optimize, lr=config.training.lr)
            warmup_steps = config.training.warmup_epoch * len(train_loader)
            total_steps = config.training.max_epoch * len(train_loader)
            scheduler = const_lr(optimizer, config.training.lr, warmup_steps, total_steps)
        else:
            optimizer = torch.optim.AdamW(params_to_optimize, lr=config.training.lr)
            lr_decay_step = config.training.lr_decay * len(train_loader)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step,
                                                        gamma=config.training.lr_decay_rate)
       
        # ==============================================================================
        # 4. INSTANCIAÇÃO DO TRAINER E EXECUÇÃO
        # ==============================================================================
        
        if config.trainer == "3dmrl_trainer":
            trainer = TrainerToMRL(
                rank=rank, config=config, model=model, logit_scale=logit_scale, 
                image_proj=image_proj, text_proj=text_proj, mrl_heads=mrl_heads,
                optimizer=optimizer, scheduler=scheduler, train_loader=train_loader,
                modelnet40_loader=modelnet40_loader, 
                objaverse_lvis_loader=objaverse_lvis_loader, scanobjectnn_loader=scanobjectnn_loader
            )

        if config.resume is not None:
            trainer.load_from_checkpoint(config.resume)

        elif config.autoresume:
            if os.path.exists(os.path.join(config.ckpt_dir, 'latest.pt')):
                trainer.load_from_checkpoint(os.path.join(config.ckpt_dir, 'latest.pt'))


        trainer.train()



    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    cli_args, extras = parse_args(sys.argv[1:])
    main(cli_args, extras)