
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, predict_calories=False, calorie_weight=1.0):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if predict_calories:
        metric_logger.add_meter('cls_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.add_meter('cal_loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    calorie_criterion = torch.nn.MSELoss()

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        if predict_calories:
            samples, targets, calories = batch
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            calories = calories.to(device, non_blocking=True).float().unsqueeze(1)
        else:
            samples, targets = batch
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            
            if predict_calories:
                class_outputs, calorie_outputs = outputs
                cls_loss = criterion(samples, class_outputs, targets)
                cal_loss = calorie_criterion(calorie_outputs, calories)
                loss = cls_loss + calorie_weight * cal_loss
            else:
                loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        if predict_calories:
            metric_logger.update(loss=loss_value)
            metric_logger.update(cls_loss=cls_loss.item())
            metric_logger.update(cal_loss=cal_loss.item())
        else:
            metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, predict_calories=False):
    criterion = torch.nn.CrossEntropyLoss()
    calorie_criterion = torch.nn.MSELoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        if predict_calories:
            images, target, calories = batch
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            calories = calories.to(device, non_blocking=True).float().unsqueeze(1)
        else:
            images, target = batch
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            
            if predict_calories:
                class_output, calorie_output = output
                loss = criterion(class_output, target)
                cal_loss = calorie_criterion(calorie_output, calories)
                
                # Calculate MAE for calories (more interpretable metric)
                cal_mae = torch.abs(calorie_output - calories).mean()
            else:
                class_output = output
                loss = criterion(class_output, target)

        acc1, acc5 = accuracy(class_output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        if predict_calories:
            metric_logger.meters['cal_loss'].update(cal_loss.item(), n=batch_size)
            metric_logger.meters['cal_mae'].update(cal_mae.item(), n=batch_size)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    if predict_calories:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} '
              'cal_mae {cal_mae.global_avg:.3f} (×1000 = {cal_mae_orig:.1f} kcal)'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss,
                      cal_mae=metric_logger.cal_mae, cal_mae_orig=metric_logger.cal_mae.global_avg * 1000))
    else:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}