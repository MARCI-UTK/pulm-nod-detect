import torch
import torch.nn as nn
import torch.nn.functional as F

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, labels, pred_locs, targets, batch_size):
        rv = 0

        loss = F.smooth_l1_loss(input=pred_locs, target=targets, beta=1, reduction='none')
        mask = torch.where(labels > 0, 1, 0)
        mask = mask.permute(0, 2, 1)

        for i in range(batch_size): 
            rv += 10 * (loss[i] * mask[i]).sum()

        return rv / 32
    

class ClsLoss(nn.Module):
    def __init__(self):
        super(ClsLoss, self).__init__()

    def forward(self, pred, targets, batch_size):
        rv = 0

        mask = torch.where(targets < 1, 0., 1.)
        loss = F.binary_cross_entropy(input=pred, target=mask, reduction='none')
        mask = torch.where(targets < 0, 0, 1)

        for i in range(batch_size): 
            rv += (1 / batch_size) * (loss[i] * mask[i]).sum()

        return rv / 32