import torch
import torch.nn as nn


class BalanceCrossEntropyLoss(nn.Module):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    '''

    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super(BalanceCrossEntropyLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                return_origin=False):
        '''
        Args:
            pred: shape :math:`(N, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
        '''
        N, H, W = gt.shape
        mask = (gt > 0).float()
        positive_count = mask.sum().item()
        negative_count = min(int(N*H*W - positive_count), int(positive_count * self.negative_ratio))

        loss = nn.functional.binary_cross_entropy_with_logits(pred, gt, reduction='none')
        positive_loss = loss * mask
        negative_loss = loss * (1. - mask)
        negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) /\
            (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss