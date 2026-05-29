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

from trainers.testado_mrltamm import CLIP_Adapter_Trainer

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


        if config.trainer == "clip_adapter_trainer":
            if rank == 0:
                logging.info("--- Iniciando Estágio 1: Treinamento MRL dos Adaptadores ---")
            
            image_adapter = NewCLIP(c_in=config.model.c_in, ratio=config.model.ratio).to(device)
            text_adapter = NewCLIP(c_in=config.model.c_in, ratio=config.model.ratio).to(device)
            
            image_proj = torch.nn.Linear(config.model.c_in, config.model.c_in).to(device)
            text_proj = torch.nn.Linear(config.model.c_in, config.model.c_in).to(device) 
            
            image_adapter = DDP(image_adapter, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            text_adapter = DDP(text_adapter, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            image_proj = DDP(image_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            text_proj = DDP(text_proj, device_ids=[rank], output_device=rank, find_unused_parameters=False)

            params_to_optimize = list(image_adapter.parameters()) + list(text_adapter.parameters()) + list(logit_scale.parameters())

            if config.training.use_image_proj:
                params_to_optimize += list(image_proj.parameters())
            if config.training.use_text_proj:
                params_to_optimize += list(text_proj.parameters())


        if config.trainer == "3dmrl_trainer":
            if rank == 0:
                logging.info("--- Starting 3DMRL trainer... ---")

            model = models.make(config).to(device)    
            if config.model.name.startswith('Mink'):
                model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
                if rank == 0: logging.info("Usando MinkowskiSyncBatchNorm no PointBERT")
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if rank == 0: logging.info("Usando SyncBatchNorm no PointBERT")

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


       # ------------------- ESTÁGIO 2: TREINAMENTO 3D DIRETO (MRL) ---------------------
        elif config.trainer == 'mrl_alignment':
            if rank == 0:
                logging.info("--- Iniciando Estágio 2: Treinamento 3D Direto com MRL ---")

            # Modelo 3D Principal (PointBERT - O Único Treinável)
            model = models.make(config).to(device)    
            if config.model.name.startswith('Mink'):
                model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
                if rank == 0: logging.info("Usando MinkowskiSyncBatchNorm no PointBERT")
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                if rank == 0: logging.info("Usando SyncBatchNorm no PointBERT")

            model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            
            if rank == 0:
                total_params = sum(p.numel() for p in model.parameters())
                logging.info(f"Network: {config.model.name}, Parâmetros de Treino: {total_params}")

            # Professores do Estágio 1 (Congelados)
            if config.get('use_pretrained', False):
                if rank == 0: 
                    logging.info(f"Carregando professores MRL de: {config.pretrained_adapters.path}")
                
                pretrained_image_adapter = NewCLIP(c_in=config.pretrained_adapters.c_in, ratio=config.pretrained_adapters.ratio).to(device)
                pretrained_text_adapter  = NewCLIP(c_in=config.pretrained_adapters.c_in, ratio=config.pretrained_adapters.ratio).to(device)

                checkpoint = torch.load(config.pretrained_adapters.path, map_location='cpu')

                img_state_dict = checkpoint.get('image_adapter', checkpoint.get('state_dict'))
                txt_state_dict = checkpoint.get('text_adapter', checkpoint.get('state_dict'))

                # 2. O TRUQUE: Remove o prefixo 'module.' de todas as chaves
                img_state_dict = {k.replace('module.', ''): v for k, v in img_state_dict.items()}
                txt_state_dict = {k.replace('module.', ''): v for k, v in txt_state_dict.items()}

                # 3. Agora sim, carrega os pesos limpinhos nos professores
                pretrained_image_adapter.load_state_dict(img_state_dict)
                pretrained_text_adapter.load_state_dict(txt_state_dict)

                # pretrained_image_adapter.eval()
                # pretrained_text_adapter.eval()
                # for param in pretrained_image_adapter.parameters():
                #     param.requires_grad = False
                # for param in pretrained_text_adapter.parameters():
                #     param.requires_grad = False

                pretrained_image_adapter = DDP(pretrained_image_adapter, device_ids=[rank], output_device=rank, find_unused_parameters=False)
                pretrained_text_adapter  = DDP(pretrained_text_adapter, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            else:
                pretrained_image_adapter = None
                pretrained_text_adapter  = None
                if rank == 0: logging.warning("Aviso: Treinando SEM os professores MRL.")

            # Parâmetros de Treino para o Estágio 2
            params_to_optimize = list(model.parameters()) + list(logit_scale.parameters())
            
        else:
            raise ValueError(f"Trainer '{config.trainer}' não reconhecido. Use 'clip_adapter_trainer' ou 'mrl_alignment'.")


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
        
        if config.trainer == "clip_adapter_trainer":
            trainer = CLIP_Adapter_Trainer(
                rank=rank, config=config,
                image_adapter=image_adapter, text_adapter=text_adapter,
                logit_scale=logit_scale, image_proj=image_proj, text_proj=text_proj,
                optimizer=optimizer, scheduler=scheduler, train_loader=train_loader
            )
            
        elif config.trainer == "3dmrl_trainer":
            trainer = TrainerToMRL(
                rank=rank, config=config, model=model, logit_scale=logit_scale, 
                image_proj=image_proj, text_proj=text_proj, mrl_heads=mrl_heads,
                optimizer=optimizer, scheduler=scheduler, train_loader=train_loader,
                modelnet40_loader=modelnet40_loader, 
                objaverse_lvis_loader=objaverse_lvis_loader, scanobjectnn_loader=scanobjectnn_loader
            )

        elif config.trainer == "mrl_alignment":
            trainer = TAMM_Trainer(
                rank=rank, config=config, model=model, logit_scale=logit_scale, 
                pretrained_image_adapter=pretrained_image_adapter, 
                pretrained_text_adapter=pretrained_text_adapter,  
                optimizer=optimizer, scheduler=scheduler, 
                train_loader=train_loader, modelnet40_loader=modelnet40_loader, 
                objaverse_lvis_loader=objaverse_lvis_loader, scanobjectnn_loader=scanobjectnn_loader
            )

        # # Retomar Treinamento se necessário
        # if config.resume is not None:
        #     trainer.load_from_checkpoint(config.resume)
        # elif config.autoresume:
        #     if os.path.exists(os.path.join(config.ckpt_dir, 'latest.pt')):
        #         trainer.load_from_checkpoint(os.path.join(config.ckpt_dir, 'latest.pt'))

        if config.resume is not None:
            trainer.load_from_checkpoint(config.resume)
            trainer.test_modelnet40()
            trainer.test_objaverse_lvis()
            trainer.test_scanobjectnn()
            

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    cli_args, extras = parse_args(sys.argv[1:])
    main(cli_args, extras)