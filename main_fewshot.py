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
from trainers.mrl_fewshot import FSL_Trainer

from utils.logger import setup_logging
from utils.misc import load_config, dump_config
from utils.scheduler import cosine_lr, const_lr
from param import parse_args

from dataset.ModelNetDataset import make_modelNet
from dataset.ModelNetDatasetFewShot import make_modelNetFewShot
# from dataset.ScanobjectNNDataset import make_ScanObjectNN, make_ScanObjectNN_hardest
# from dataset.ObjaverseLVIS import make_objaverse_lvis

from trainers.MRL import MRL_Linear_Heads
from trainers.MRL import MRL_Projection_Layer


def build_dataloader(config):
    if config.dataset.NAME == "ModelNet":
        train_loader = make_modelNet(config.dataset)
        test_loader = make_modelNet(config.test_dataset)

    elif config.dataset.NAME == "ModelNetFewShot":
        train_loader = make_modelNetFewShot(config.dataset)
        test_loader = make_modelNetFewShot(config.test_dataset)

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
    if config.trainer == "few-shot":
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

        # 2. Instanciar e Carregar pesos do MRL_HEADS (Projeção pré-treinada)
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
        raise ValueError(f"Unknown trainer: {config.trainer}")


    # ==========================================================
    # 10-TRIAL FEW-SHOT LOOP
    # ==========================================================
    mrl_dims_strs = [str(dim) for dim in config.linear_layer.mrl_dims]
    all_trials_overall_acc = {dim: [] for dim in mrl_dims_strs}
    all_trials_class_acc = {dim: [] for dim in mrl_dims_strs}
    
    num_trials = 10 
    base_seed = config.seed if config.fix_seed else 42 # Starting seed

    # CRITICAL FIX: Ensure linear layer output matches the N-Way task (e.g., 5 instead of 40)
    if config.dataset.NAME == "ModelNetFewShot":
        config.linear_layer.out_dim = config.dataset.way

    for trial in range(num_trials):
        # 1. Update the seed for this specific trial
        current_seed = base_seed + trial
        
        random.seed(current_seed + rank)
        np.random.seed(current_seed + rank)
        torch.manual_seed(current_seed + rank)
        torch.cuda.manual_seed_all(current_seed + rank)

        if rank == 0:
            logging.info(f"\n{'='*65}\nStarting Few-Shot Trial {trial + 1}/{num_trials} (Fold {trial} | Seed {current_seed})\n{'='*65}")

        # 2. Update Config to load the correct benchmark .pkl file
        config.dataset.fold = trial
        config.test_dataset.fold = trial

        # 3. Build DataLoaders for this fold
        train_loader, test_loader = build_dataloader(config)

        # 4. Initialize fresh Linear Heads for this trial
        # IMPORTANTE: Apenas esta camada será destreinada/atualizada pelo otimizador
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

        # 5. Initialize fresh Optimizer
        lr = config.training.lr 
        optimizer = torch.optim.AdamW(
            linear_layer.parameters(), # <-- Passando SOMENTE os pesos da linear layer!
            lr=lr,
            weight_decay=config.training.weight_decay,
        )

        # 6. Initialize fresh Scheduler
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

        # 7. Initialize Trainer
        trainer = FSL_Trainer(
            rank=rank,
            config=config,
            model=model,
            mrl_heads=mrl_heads,         # <-- Passando as MRL heads (congeladas) para o trainer
            linear_layer=linear_layer,   # <-- Passando a Linear Head (treinável) para o trainer
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            test_loader=test_loader,
        )

        # (Optional) Resume logic applies to the current fold
        if config.resume:
            trainer.load_from_checkpoint(config.resume)

        # 8. Train and retrieve best metrics
        best_oa, best_ca = trainer.train()

        # 9. Aggregate metrics
        for dim in mrl_dims_strs:
            all_trials_overall_acc[dim].append(best_oa[dim])
            all_trials_class_acc[dim].append(best_ca[dim])
            
        # Ensure all GPUs wait before starting the next fold
        dist.barrier()

    # ==========================================================
    # FINAL RESULTS (MEAN ± STD)
    # ==========================================================
    if rank == 0:
        logging.info("\n" + "="*85)
        logging.info(f"FINAL 10-TRIAL FEW-SHOT RESULTS ({config.dataset.way}-Way {config.dataset.shot}-Shot)")
        logging.info("="*85)
        
        header = f"| {'Dim':>6} | {'Overall Acc (Mean ± Std)':>28} | {'Class Acc (Mean ± Std)':>28} |"
        logging.info(header)
        logging.info("-" * len(header))
        
        for dim in mrl_dims_strs:
            oa_arr = np.array(all_trials_overall_acc[dim]) * 100
            ca_arr = np.array(all_trials_class_acc[dim]) * 100
            
            oa_mean, oa_std = oa_arr.mean(), oa_arr.std()
            ca_mean, ca_std = ca_arr.mean(), ca_arr.std()
            
            row = f"| {dim:>6} | {oa_mean:>20.2f}% ± {oa_std:<5.2f} | {ca_mean:>20.2f}% ± {ca_std:<5.2f} |"
            logging.info(row)
            
        logging.info("-" * len(header))

    dist.destroy_process_group()


if __name__ == "__main__":
    cli_args, extras = parse_args(sys.argv[1:])
    main(cli_args, extras)