from __future__ import absolute_import

import torch
import torch.nn.functional as F

from .seqCrossEntropyLoss import SeqCrossEntropyLoss
from .seqLabelSmoothingCrossEntropyLoss import SeqLabelSmoothingCrossEntropyLoss
from .balance_cross_entropy_loss import BalanceCrossEntropyLoss

def DiceLoss(x_o, x_t):
    x_o = torch.sigmoid(x_o)
    iflat = x_o.view(-1)
    tflat = x_t.view(-1)
    intersection = (iflat * tflat).sum()
    return 1. - torch.mean((2. * intersection + 1e-5) / (iflat.sum() + tflat.sum() + 1e-5))

def DiceLossNoSigmoid(x_o, x_t):
    iflat = x_o.reshape(-1)
    tflat = x_t.reshape(-1)
    intersection = (iflat * tflat).sum()
    return 1. - torch.mean((2. * intersection + 1e-5) / (iflat.sum() + tflat.sum() + 1e-5))

def MultiClassDiceLoss(x_o, x_t):
    B, num_cls, H, W = x_o.shape
    x_o = torch.sigmoid(x_o)
    # [B, H, W, num_cls] -> [B, num_cls, H, W]
    x_t = x_t.permute(0, 3, 1, 2)
    iflat = x_o.reshape(-1)
    tflat = x_t.reshape(-1)
    intersection = (iflat * tflat).sum()
    return 1. - torch.mean((2. * intersection + 1e-5) / (iflat.sum() + tflat.sum() + 1e-5))

def CrossEntropyDiceLoss(x_o, x_t):
    # x_o: [B, num_cls, H, W]
    # x_t: [B, H, W, num_cls-1], no background
    B, num_cls, H, W = x_o.shape
    ## generate gt: expand with background
    onehot_gt = torch.zeros((B, H, W, num_cls)).type_as(x_t)
    x_t[x_t>0.5] = 1
    x_t[x_t<=0.5] = 0
    fg_x_t = x_t.sum(-1) # [B, H, W]
    onehot_gt[:, :, :, :-1] = x_t
    onehot_gt[:, :, :, -1]  = fg_x_t == 0 # background
    onehot_gt = onehot_gt.permute(0, 3, 1, 2) # [B, num_cls, H, W]
    ## predict
    x_o = F.softmax(x_o, dim=1)
    ## loss
    ### loss weight
    weight = torch.ones(num_cls).type_as(x_o) # the weight of bg is 1.
    num_bg = (fg_x_t == 0).sum()
    weight[:-1] = num_bg / (B*H*W - num_bg)

    total_loss = 0
    for i in range(onehot_gt.shape[1]):
        dice_loss = DiceLossNoSigmoid(x_o[:,i], onehot_gt[:,i])
        dice_loss *= weight[i]
        total_loss += dice_loss

    return total_loss/weight.sum()

def BanlanceMultiClassCrossEntropyLoss(x_o, x_t):
    # [B, num_cls, H, W]
    B, num_cls, H, W = x_o.shape
    x_o = x_o.reshape(B, num_cls, H*W).permute(0, 2, 1)
    # [B, H, W, num_cls]
    ## generate gt
    x_t[x_t>0.5] = 1
    x_t[x_t<=0.5] = 0
    fg_x_t = x_t.sum(-1) # [B, H, W]
    x_t = x_t.argmax(-1) # [B, H, W]
    x_t[fg_x_t==0] = num_cls - 1 # background
    x_t = x_t.reshape(B, H*W)
    # loss
    weight = torch.ones((B, num_cls)).type_as(x_o) # the weight of bg is 1.
    num_bg = (x_t == (num_cls - 1)).sum(-1) # [B]
    weight[:, :-1] = (num_bg / (H*W - num_bg+1e-5)).unsqueeze(-1).expand(-1, num_cls-1)
    logit = F.log_softmax(x_o, dim=-1) # [B, H*W, num_cls]
    logit = logit * weight.unsqueeze(1)
    loss = - logit.gather(2, x_t.unsqueeze(-1).long())
    return loss.mean()