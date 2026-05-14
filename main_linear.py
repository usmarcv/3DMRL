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

import models
from trainers.mrl_linear_probing_trainer import Linear_Probing_Trainer 

from utils.logger import setup_logging
from utils.misc import load_config, dump_config
from utils.scheduler import cosine_lr, const_lr
from param import parse_args

from dataset.ModelNetDataset import make_modelNet
# from dataset.ScanobjectNNDataset import make_ScanObjectNN, make_ScanObjectNN_hardest
# from dataset.ObjaverseLVIS import make_objaverse_lvis

from trainers.MRL import MRL_Linear_Heads
from trainers.MRL import MRL_Projection_Layer


def build_dataloader(config):
    if config.dataset.NAME == "ModelNet":
        train_loader = make_modelNet(config.dataset)
        test_loader = make_modelNet(config.test_dataset)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset.NAME}")

    return train_loader, test_loader


def main(cli_args, extras):
    # ==========================================================
    # INIT DDP & LOGGING (Run once)
    # ==========================================================
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
        config.trial_name += "@autoresume"
    else:
        config.trial_name += datetime.now().strftime("@%Y%m%d-%H%M%S")

    config.ckpt_dir = config.get("ckpt_dir") or os.path.join(
        config.exp_dir, config.trial_name, "ckpt"
    )
    config.code_dir = config.get("code_dir") or os.path.join(
        config.exp_dir, config.trial_name, "code"
    )

    config.ngpu = world_size

    if rank == 0:
        os.makedirs(os.path.join(config.exp_dir, config.trial_name), exist_ok=config.autoresume)
        os.makedirs(config.ckpt_dir, exist_ok=True)

        if os.path.exists(config.code_dir):
            shutil.rmtree(config.code_dir)

        config.log_path = config.get("log_path") or os.path.join(
            config.exp_dir, config.trial_name, "log.txt"
        )

        config.log_level = logging.DEBUG if config.debug else logging.INFO
        setup_logging(config.log_path, config.log_level)

        dump_config(os.path.join(config.exp_dir, config.trial_name, "config.yaml"), config)

    # ==========================================================
    # FIX SEED
    # ==========================================================
    if config.fix_seed:
        random.seed(config.seed + rank)
        np.random.seed(config.seed + rank)
        torch.manual_seed(config.seed + rank)
        torch.cuda.manual_seed_all(config.seed + rank)

    # ==========================================================
    # MODEL (3D BACKBONE) - Initialize & Load Once
    # ==========================================================
    model = models.make(config).cuda(rank)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Network: {config.model.name}")
        logging.info(f"Parameters: {total_params}")

    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
    )

    if config.model.name.startswith("Mink"):
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
        if rank == 0: logging.info("Using MinkowskiSyncBatchNorm")
    else:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if rank == 0: logging.info("Using SyncBatchNorm")

    # ==========================================================
    # CARREGAMENTO E CONGELAMENTO PARA LINEAR PROBING
    # ==========================================================
    if config.trainer == "linear_probing":
        checkpoint = torch.load(config.pretrained_model.path, map_location="cpu")
        state_dict_original = checkpoint["state_dict"]
        
        state_dict_corrigido = {}
        
        # Passamos por todas as chaves adicionando o "module." na frente
        for key, value in state_dict_original.items():
            new_key = f"module.{key}" if not key.startswith("module.") else key
            state_dict_corrigido[new_key] = value
            
        model.load_state_dict(state_dict_corrigido)
        # Congelar totalmente o Backbone
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        if rank == 0: logging.info("Backbone congelado com sucesso para Linear Probing.")

        # Instanciar e Carregar pesos do MRL_HEADS (Projeção pré-treinada)
        mrl_heads = MRL_Projection_Layer(
            nesting_list=config.mrl.nesting_dims,
            out_dim=config.mrl.out_dim,
            efficient=config.mrl.efficient
        ).cuda(rank)

        mrl_heads = DDP(
            mrl_heads, 
            device_ids=[rank], 
            output_device=rank, 
            find_unused_parameters=False
        )

        # Tratar chaves da MRL Head
        mrl_state_dict_original = checkpoint["mrl_heads"]
        mrl_state_dict_corrigido = {}
        for key, value in mrl_state_dict_original.items():
            new_key = f"module.{key}" if not key.startswith("module.") else key
            mrl_state_dict_corrigido[new_key] = value
            
        mrl_heads.load_state_dict(mrl_state_dict_corrigido)

        # Congelar totalmente as cabeças de projeção MRL
        mrl_heads.eval()
        for param in mrl_heads.parameters():
            param.requires_grad = False
            
        if rank == 0: logging.info("MRL Heads carregadas e congeladas com sucesso.")

    else:
        raise ValueError(f"Unknown trainer configuration: {config.trainer}. Expected 'linear-probing'")


    # ==========================================================
    # DATALOADERS & LINEAR HEAD SETUP
    # ==========================================================
    train_loader, test_loader = build_dataloader(config)

    # Inicializa a camada linear (Esta é a única camada que será treinada)
    linear_layer = MRL_Linear_Heads(
        mrl_dims=config.linear_layer.mrl_dims, 
        out_dim=config.linear_layer.out_dim
    ).cuda(rank)

    linear_layer = DDP(
        linear_layer,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False,
    )

    # Otimizador focado apenas na camada linear
    lr = config.training.lr 
    optimizer = torch.optim.AdamW(
        linear_layer.parameters(),
        lr=lr,
        weight_decay=config.training.weight_decay,
    )

    # Scheduler
    warmup = config.training.warmup_epoch * len(train_loader)
    total_steps = config.training.max_epoch * len(train_loader)

    if config.training.scheduler == "cosine":
        scheduler = cosine_lr(optimizer, lr, warmup, total_steps)
    elif config.training.scheduler == "const":
        scheduler = const_lr(optimizer, lr, warmup, total_steps)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.lr_decay * len(train_loader),
            gamma=config.training.lr_decay_rate,
        )

    # ==========================================================
    # TREINAMENTO LINEAR PROBING
    # ==========================================================
    trainer = Linear_Probing_Trainer(
        rank=rank,
        config=config,
        model=model,
        mrl_heads=mrl_heads,  #frozen
        linear_layer=linear_layer,  # trainable layer
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    if config.resume:
        trainer.load_from_checkpoint(config.resume)

    if rank == 0:
        logging.info(f"\n{'='*65}\nStarting Linear Probing Training\n{'='*65}")

    # Treina o modelo e recupera as métricas
    best_oa, best_ca = trainer.train()

    # ==========================================================
    # RESULTADOS FINAIS
    # ==========================================================
    if rank == 0:
        logging.info("\n" + "="*85)
        logging.info("FINAL LINEAR PROBING RESULTS")
        logging.info("="*85)
        
        mrl_dims_strs = [str(dim) for dim in config.linear_layer.mrl_dims]
        header = f"| {'Dim':>6} | {'Overall Accuracy':>20} | {'Class Accuracy':>20} |"
        logging.info(header)
        logging.info("-" * len(header))
        
        for dim in mrl_dims_strs:
            oa_val = best_oa[dim] * 100
            ca_val = best_ca[dim] * 100
            row = f"| {dim:>6} | {oa_val:>19.2f}% | {ca_val:>19.2f}% |"
            logging.info(row)
            
        logging.info("-" * len(header))

    dist.destroy_process_group()


if __name__ == "__main__":
    cli_args, extras = parse_args(sys.argv[1:])
    main(cli_args, extras)