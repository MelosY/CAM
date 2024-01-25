# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Iterable, Optional
import sys

import torch
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from utils import adjust_learning_rate

from loss import *
import evaluation_metric

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None, data_loader_val=None, max_accuracy=0.):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    for data_iter_step, (samples, targets, tgt_lens, binary_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # During training, evaluation is conducted, therefore, the training flag should be reset again.
        model.train(True)
        

        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # add binary supervision
        
        binary_mask = binary_mask.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model((samples, targets, tgt_lens, binary_mask))
                loss = criterion(output['rec_output'], targets, tgt_lens)
                metric_logger.update(loss_recog=loss.item())


                loss_binary = eval(args.binary_loss_type)(output['pred_binary'], binary_mask)
                if not args.discard_dice_loss:
                    loss_binary += \
                        DiceLoss(output['pred_binary'].sum(1), binary_mask.sum(-1))
                # loss_binary = MultiClassDiceLoss(output['pred_binary'], binary_mask) + \

                metric_logger.update(loss_binary=loss_binary.item())
                loss += args.loss_weight_binary * loss_binary

        else: # full precision
            output = model((samples, targets, tgt_lens, binary_mask))
            loss = criterion(output['rec_output'], targets, tgt_lens)
            metric_logger.update(loss_recog=loss.item())


            loss_binary = eval(args.binary_loss_type)(output['pred_binary'], binary_mask)
            if not args.discard_dice_loss:
                loss_binary += \
                    DiceLoss(output['pred_binary'].sum(1), binary_mask.sum(-1))
            # loss_binary = MultiClassDiceLoss(output['pred_binary'], binary_mask) + \

            metric_logger.update(loss_binary=loss_binary.item())
            loss += args.loss_weight_binary * loss_binary




        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(targets)
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)
       
        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        
        torch.cuda.synchronize()

        if mixup_fn is None:
            # preds = F.softmax(output['rec_output'], dim=-1)
            # _, pred_ids = preds.max(-1)
            # class_acc = evaluation_metric.factory()['accuracy'](pred_ids, targets, data_loader.dataset)
            # for Chinese, the above evaluation is a little time-consuming.
            class_acc = 0.
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        # evaluation during training
        if data_iter_step >= 1 and data_iter_step % args.eval_freq == 0:
            if data_loader_val is not None:
                test_stats = evaluate(data_loader_val, model, device, args=args)
                print(f"Accuracy of the network on the {len(data_loader_val.dataset)} test images: {test_stats['acc']:.4f}%")
                if max_accuracy < test_stats["acc"]:
                    max_accuracy = test_stats["acc"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)

        if data_iter_step >= 1 and data_iter_step % (args.eval_freq * 10) == 0:
            utils.save_model(
                args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch="{0}_{1}".format(epoch, data_iter_step), model_ema=model_ema)
        elif epoch >=5 and data_iter_step % (args.eval_freq) == 0:
            utils.save_model(
                args=args, model=model, model_without_ddp=model.module, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch="{0}_{1}".format(epoch, data_iter_step), model_ema=model_ema)
        # flush the screen info to disk_file.
        # if utils.is_main_process():
        sys.stdout.flush()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    train_stats.update({'max_accuracy': max_accuracy})
    return train_stats

@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False, args=None):
    criterion = SeqCrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[1]
        lens = batch[2]
        binary_feat = batch[3]


        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        binary_feat = binary_feat.to(device, non_blocking=True)
        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model((images, target, lens, None,None))
                if isinstance(output, dict):
                    output = output['rec_output']
                if args.beam_width > 0:
                    loss = torch.Tensor([0.])
                else:
                    loss = criterion(output, target, lens)
        else:
            output = model((images, target, lens, binary_feat))
            if isinstance(output, dict):
                output = output['rec_output']
            if args.beam_width > 0:
                loss = torch.Tensor([0.])
            else:
                loss = criterion(output, target, lens)

        torch.cuda.synchronize()
        
        # evaluation metrics.
        if args.beam_width > 0:
            pred_ids = output
        else:
            _, pred_ids = output.max(-1)
        
        #ipdb.set_trace()
        acc = evaluation_metric.factory()['accuracy'](pred_ids, target, data_loader.dataset)
        recognition_fmeasure = evaluation_metric.factory()['recognition_fmeasure'](pred_ids, target, data_loader.dataset)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc'].update(acc, n=batch_size)
        metric_logger.meters['recognition_fmeasure'].update(recognition_fmeasure, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* {eval_data.root}: {acc.count} images, Acc {acc.global_avg:.4f} loss {losses.global_avg:.4f} Rec_fmeasure {rec_f.global_avg:.4f}'
          .format(eval_data=data_loader.dataset, acc=metric_logger.acc, losses=metric_logger.loss, rec_f=metric_logger.recognition_fmeasure))
    # the window size of smoothedvalue is set to 20, therefore there may be imprecise.
    if len(metric_logger.meters['acc'].deque) == metric_logger.meters['acc'].window_size:
        print('there are too many batches, therefore this accuracy may be not accurate.')

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}