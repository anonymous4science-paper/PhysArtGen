import math
import sys
from typing import Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = None,
                    log_writer=None,
                    args=None, ae=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # Adjust learning rate at the end of each accumulation
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # Unpack data - Three ID data format
        if len(batch) == 7:  # Contains surface point cloud
            samples, labels, surface, category_id, major_category_id, part_category_id, identifier = batch
        else:  # Does not contain surface point cloud
            samples, labels, category_id, major_category_id, part_category_id, identifier = batch
            surface = None

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if surface is not None:
            surface = surface.to(device, non_blocking=True)
        
        # Transfer three types of IDs to device
        major_category_id = major_category_id.to(device, non_blocking=True)
        part_category_id = part_category_id.to(device, non_blocking=True)
        
        # Process identifier - can be string or tensor
        if isinstance(identifier[0], str):
            # If it's a string list, pass it to the model as is
            identifier_input = identifier
        else:
            # If it's already a tensor, transfer to device
            identifier_input = identifier.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            # Use AutoEncoder to encode surface point cloud to latent representation
            with torch.no_grad():
                _, encoded_surface = ae.encode(surface)
            
            # Fix: Use EDMLoss to correctly generate noise and calculate loss (consistent with original project)
            from models_three_id_fixed import EDMLoss
            if not hasattr(train_one_epoch, 'edm_loss'):
                train_one_epoch.edm_loss = EDMLoss(P_mean=-1.2, P_std=1.2, sigma_data=1)
            
            # Use standard EDM loss calculation
            loss = train_one_epoch.edm_loss(
                lambda x, sigma, labels: model(
                    x, sigma,
                    major_category_ids=major_category_id,
                    part_category_ids=part_category_id,
                    identifiers=identifier_input
                ),
                encoded_surface,
                labels=None  # Our conditional information is passed through three IDs
            )

        loss_value = loss.item()

        # Enhanced numerical stability check
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(f"Debug info - encoded_surface range: [{encoded_surface.min().item():.4f}, {encoded_surface.max().item():.4f}]")
            # Add model parameter check
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN detected in parameter: {name}")
                if torch.isinf(param).any():
                    print(f"Inf detected in parameter: {name}")
            sys.exit(1)
            
        # Add gradient explosion check
        if loss_value > 100.0:
            print(f"Warning: Very large loss detected: {loss_value}")

        loss /= accum_iter
        
        # Distributed training safe gradient update
        try:
            loss_scaler(loss, optimizer, clip_grad=clip_grad,
                        parameters=model.parameters(), create_graph=False,
                        update_grad=(data_iter_step + 1) % accum_iter == 0)
        except RuntimeError as e:
            print(f"Error in gradient scaling/backward: {e}")
            print(f"Loss value: {loss_value}, batch size: {surface.shape[0]}")
            # Skip this batch instead of crashing
            if "monitoredBarrier" in str(e) or "timeout" in str(e).lower():
                print("Distributed training timeout detected, skipping batch...")
                return {'loss': loss_value}
            else:
                raise e
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # Collect statistics
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, ae=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        # Unpack data - Three ID data format
        if len(batch) == 7:  # Contains surface point cloud
            samples, labels, surface, category_id, major_category_id, part_category_id, identifier = batch
        else:  # Does not contain surface point cloud
            samples, labels, category_id, major_category_id, part_category_id, identifier = batch
            surface = None

        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if surface is not None:
            surface = surface.to(device, non_blocking=True)
        
        # Transfer three types of IDs to device
        major_category_id = major_category_id.to(device, non_blocking=True)
        part_category_id = part_category_id.to(device, non_blocking=True)
        
        # Process identifier
        if isinstance(identifier[0], str):
            identifier_input = identifier
        else:
            identifier_input = identifier.to(device, non_blocking=True)

        # Calculate output
        with torch.amp.autocast('cuda'):
            # Use AutoEncoder to encode surface point cloud
            _, encoded_surface = ae.encode(surface)
            
            # For evaluation, we can test the model's reconstruction capability
            # Add small amount of noise then denoise
            batch_size = encoded_surface.shape[0]
            sigma = torch.ones(batch_size, device=device) * 5.0  # Fixed small noise level
            
            noise = torch.randn_like(encoded_surface)
            noisy_encoded = encoded_surface + sigma.reshape(-1, 1, 1) * noise
            
            # Model prediction
            predictions = model(
                noisy_encoded,
                sigma,
                major_category_ids=major_category_id,
                part_category_ids=part_category_id,
                identifiers=identifier_input
            )
            
            # Reconstruction loss
            loss = F.mse_loss(predictions, encoded_surface)

        # Simple "accuracy" metric: based on reconstruction error
        reconstruction_error = F.mse_loss(predictions, encoded_surface, reduction='none').mean(dim=[1,2])
        # If reconstruction error is below threshold, consider it "correct"
        threshold = 0.1
        acc1 = (reconstruction_error < threshold).float().mean() * 100

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    # Collect statistics
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['acc1'], losses=metric_logger.meters['loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} 