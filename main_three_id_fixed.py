#!/usr/bin/env python3

import argparse
import datetime
import os
import time
import json
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    print("Warning: TensorBoard not installed, skipping logging")
    SummaryWriter = None
    HAS_TENSORBOARD = False
import numpy as np

# Import fixed models
import models_three_id_fixed
import models_ae
from util.drawer_dataset import DrawerDataset
from engine_three_id import train_one_epoch, evaluate
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import datetime

def get_args_parser():
    parser = argparse.ArgumentParser('Three ID Conditional 3D Generation', add_help=False)
    
    # Basic parameters
    parser.add_argument('--batch_size', default=128, type=int, help='Training batch size')
    parser.add_argument('--epochs', default=200, type=int, help='Number of training epochs')
    parser.add_argument('--save_freq', default=50, type=int, help='Save checkpoint frequency (epochs)')
    
    # Model parameters
    parser.add_argument('--model', default='three_id_kl_d512_m512_l8_edm', type=str, help='Model name')
    parser.add_argument('--ae', default='kl_d512_m512_l8', type=str, help='AutoEncoder model')
    parser.add_argument('--ae_pth', required=True, type=str, help='AutoEncoder weights path')
    
    # Data parameters
    parser.add_argument('--data_path', required=True, type=str, help='Data path')
    parser.add_argument('--identifier_mapping_file', type=str, 
                       default='/home/zhaochaoyang/yuantingyu/3DShape2vecset/data_drawer/identifier_mapping.json',
                       help='Identifier mapping file path')
    
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW beta2')
    parser.add_argument('--accum_iter', type=int, default=1, help='Gradient accumulation steps')
    
    # Learning rate scheduling parameters
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--layer_decay', type=float, default=0.75, help='Layer decay rate')
    parser.add_argument('--clip_grad', type=float, default=None, help='Gradient clipping threshold')
    
    # Output and logging
    parser.add_argument('--output_dir', default='./output_three_id_fixed', help='Output directory')
    parser.add_argument('--log_dir', default=None, help='TensorBoard log directory')
    parser.add_argument('--device', default='cuda', help='Training device')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--resume', default='', help='Resume training path')
    
    # Distributed training
    parser.add_argument('--distributed', action='store_true', help='Whether to use distributed training')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='Distributed backend')
    parser.add_argument('--dist_url', default='env://', help='Distributed URL')
    parser.add_argument('--dist_on_itp', action='store_true', help='Whether to run distributed training on ITP cluster')
    parser.add_argument('--world_size', default=1, type=int, help='Total number of processes')
    parser.add_argument('--rank', default=0, type=int, help='Global rank')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID')
    
    # Validation
    parser.add_argument('--eval_freq', default=10, type=int, help='Validation frequency')
    
    return parser

def main(args):
    # Handle distributed training - Fix: unconditional initialization (consistent with original project)
    misc.init_distributed_mode(args)
    
    print(f'Job directory: {os.getcwd()}')
    
    device = torch.device(args.device)
    
    # Set random seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    if misc.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        if args.log_dir is None:
            args.log_dir = args.output_dir
        os.makedirs(args.log_dir, exist_ok=True)
    
    # Create dataset
    print("Creating dataset...")
    dataset_train = DrawerDataset(
        dataset_folder=args.data_path,
        split='train',
        categories=None,  # Automatically discover all categories
        pc_size=2048,
        return_surface=True,
        replica=1  # Fix: don't use online augmentation since offline augmentation was already done
    )
    
    print(f'Training set: {len(dataset_train)} samples')
    
    # Create validation set (extract a small portion from training set)
    val_size = min(100, len(dataset_train) // 10)
    if val_size > 0:
        val_indices = np.random.choice(len(dataset_train), val_size, replace=False)
        dataset_val = Subset(dataset_train, val_indices)
        print(f'Validation set: {len(dataset_val)} samples')
    else:
        dataset_val = None
        print('Validation set: 0 samples (skipping validation)')
    
    # Create data loaders - Fix: consistent distributed configuration with original project
    if True:  # Always use distributed mode (consistent with original project)
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        
        if dataset_val:
            # Validation set also uses DistributedSampler, shuffle=True (consistent with original project)
            sampler_val = DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_val = None
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) if dataset_val else None
    
    data_loader_train = DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        shuffle=(sampler_train is None)
    )
    
    if dataset_val and len(dataset_val) > 0:
        val_batch_size = min(args.batch_size, len(dataset_val))
        data_loader_val = DataLoader(
            dataset_val,
            sampler=sampler_val, 
            batch_size=val_batch_size,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )
    else:
        data_loader_val = None
    
    # Create model
    print("Creating model...")
    if args.model == 'three_id_kl_d512_m512_l8_edm':
        model = models_three_id_fixed.three_id_kl_d512_m512_l8_edm(
            identifier_mapping_file=args.identifier_mapping_file
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Load AutoEncoder
    print("Loading AutoEncoder...")
    if args.ae == 'kl_d512_m512_l8':
        ae = models_ae.KLAutoEncoder(
            dim=512, 
            num_latents=512, 
            latent_dim=8, 
            num_inputs=2048
        )
    else:
        raise ValueError(f"Unknown AE model: {args.ae}")
    
    # Load AE weights
    ae_checkpoint = torch.load(args.ae_pth, map_location='cpu')
    ae.load_state_dict(ae_checkpoint['model'])
    ae.to(device)
    ae.eval()  # Keep AE in evaluation mode
    
    print(f"AE parameters: {sum(p.numel() for p in ae.parameters()) / 1e6:.2f}M")
    
    # Distributed training setup
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)
        model_without_ddp = model.module
    
    # Optimizer
    param_groups = [{'params': model_without_ddp.parameters()}]
    optimizer = torch.optim.AdamW(
        param_groups, 
        lr=args.lr, 
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )
    
    loss_scaler = NativeScaler()
    
    # TensorBoard
    if misc.is_main_process() and HAS_TENSORBOARD:
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None
    
    # Resume training
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        print(f"Resuming training from epoch {start_epoch}")
    
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if args.distributed and sampler_train is not None:
            sampler_train.set_epoch(epoch)
        
        # Training
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler, 
            log_writer=writer, args=args, ae=ae
        )
        
        # Validation
        if data_loader_val and (epoch % args.eval_freq == 0 or epoch == args.epochs - 1):
            val_stats = evaluate(data_loader_val, model, device, ae)
            if misc.is_main_process() and writer:
                writer.add_scalar('val/loss', val_stats['loss'], epoch)
        else:
            val_stats = {}
        
        # Save checkpoint - Fix: consistent with original project, remove extra barrier
        if args.output_dir and ((epoch % args.save_freq == 0 and epoch > 0) or epoch == args.epochs - 1):
            misc.save_model(
                args=args, epoch=epoch, model=model, model_without_ddp=model_without_ddp, 
                optimizer=optimizer, loss_scaler=loss_scaler
            )
            print(f"Saved checkpoint: epoch {epoch}")
        
        # Logging
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}
        
        if misc.is_main_process():
            print(f"Epoch {epoch}: train_loss={train_stats.get('loss', 0):.4f}, "
                  f"val_loss={val_stats.get('loss', 'N/A')}")
            
            # Save log
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training completed, total time: {total_time_str}')
    
    if misc.is_main_process() and writer:
        writer.close()

if __name__ == '__main__':
    # Handle command line argument compatibility
    if '--local-rank' in sys.argv:
        # Convert --local-rank to --local_rank
        for i, arg in enumerate(sys.argv):
            if arg.startswith('--local-rank='):
                sys.argv[i] = arg.replace('--local-rank=', '--local_rank=')
            elif arg == '--local-rank':
                sys.argv[i] = '--local_rank'
    
    args = get_args_parser()
    args = args.parse_args()
    
    # Auto-detect distributed training
    if 'LOCAL_RANK' in os.environ:
        args.distributed = True
        args.local_rank = int(os.environ['LOCAL_RANK'])
    
    main(args) 