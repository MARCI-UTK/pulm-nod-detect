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

        for i in range(len(loss)): 
            rv += 20 * (loss[i] * mask[i]).sum()

        return rv / len(loss)
    

class ClsLoss(nn.Module):
    def __init__(self):
        super(ClsLoss, self).__init__()

    def forward(self, pred, targets, batch_size):
        rv = 0

        pos_weight = (targets == 0.).sum() / (targets == 1.).sum()
        loss = F.binary_cross_entropy_with_logits(input=pred, target=targets, pos_weight=pos_weight, reduction='none')

        for i in range(len(loss)): 
            rv += (1. / 32) * loss[i].sum()

        return rv / len(loss)
    
class ValClsLoss(nn.Module):
    def __init__(self):
        super(ValClsLoss, self).__init__()

    def forward(self, pred, targets, batch_size):
        rv = 0

        if (targets == 1.).sum() == 0: 
            pos_weight = (targets == 0).sum()
        else: 
            pos_weight = (targets == 0.).sum() / (targets == 1.).sum()
            
        weight = targets * (targets < 0).float()

        loss = F.binary_cross_entropy_with_logits(input=pred, target=targets, pos_weight=pos_weight, weight=weight, reduction='none')
        mask = torch.where(targets < 0, 0, 1)

        for i in range(len(loss)): 
            rv += (1. / 32) * (loss[i] * mask[i]).sum()

        return rv / len(loss)