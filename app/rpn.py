import numpy as np 
import itertools
import os
from src.util.util import get_iou, xyzd_2_2corners, corners_2_xyzd
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
import torch.nn as nn 
import torch
import torch.nn.functional as F

from src.model.feature_extractor import FeatureExtractor
from src.util.util import get_pos_weight_val, sample_anchor_boxes

# Output of Nature paper before RPN is 24x24x24x128

class RPN(nn.Module): 
    def __init__(self, in_channels: int, mid_channels: int, n_anchor: int): 
        super(RPN, self).__init__()

        self.conv1    = nn.Conv3d(in_channels=in_channels, out_channels=512, 
                                  kernel_size=3, stride=1, padding=1)
        
        self.conv_cls = nn.Conv3d(in_channels=512, out_channels=n_anchor, kernel_size=1, stride=1)
        self.conv_reg = nn.Conv3d(in_channels=512, out_channels=n_anchor * 4, kernel_size=1, stride=1)

        # Initialize weights and biases as described in Faster-RCNN paper
        # Originally, these were the normal distribution from 0-0.01, but Dr. Santos suggested 
        # increasing to 0.1 to see what happens
        """
        self.conv1.weight.data.normal_(0, 0.1)
        self.conv1.bias.data.zero_()

        self.conv_cls.weight.data.normal_(0, 0.1)
        self.conv_cls.bias.data.zero_()

        self.conv_reg.weight.data.normal_(0, 0.1)
        self.conv_reg.bias.data.zero_() 
        """
    
    def forward(self, x): 
     
        x = self.conv1(x)

        pred_anc_locs = self.conv_reg(x)
        pred_cls_scores = self.conv_cls(x)

        pred_anc_locs = pred_anc_locs.permute(0, 2, 3, 4, 1).contiguous()
        pred_anc_locs = pred_anc_locs.view(pred_anc_locs.shape[0], 24 * 24 * 24 * 3, 4)

        pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 4, 1).contiguous()
        pred_cls_scores = pred_cls_scores.view(pred_cls_scores.shape[0], 1, 24 * 24 * 24 * 3)

        # Sigmoid needed when training ROI, not needed when training RPN because of loss w/ logits function
        # pred_cls_scores = torch.sigmoid(pred_cls_scores)

        return pred_anc_locs, pred_cls_scores

def get_centers(orig_width, feat_width): 
    # This array is 24 centers (one for each FM pixel) that map to the the 4x4 slice  
    # of original image represented by FM pixel 
    centers = np.arange(2, 96, 4)

    centers = np.array(list(itertools.product(centers, repeat=3)))

    return centers

def get_anc_boxes(centers): 
    # Defined to be these values in Nature paper
    box_sizes = [5, 10, 20]

    anchor_boxes = []
    for c in centers: 
        for s in box_sizes: 
            tmp = np.append(c, s)
            corners = xyzd_2_2corners(tmp)
            anchor_boxes.append(corners)

    return anchor_boxes

def rpn_iteration(data, feature_extractor, rpn): 
    fname, x, y, bb_y = data

    # Make labels correct size 
    y = y.unsqueeze(1)

    x = x.to(f'cuda:{feature_extractor.device_ids[0]}')
    y = y.to(f'cuda:{rpn.device_ids[0]}')
    bb_y = bb_y.to(f'cuda:{rpn.device_ids[0]}')
                
    fm = feature_extractor(x)
    pred_anch_locs, pred_cls_scores = rpn(fm)

    sampled_pred, sampled_target = sample_anchor_boxes(pred_cls_scores, y)

    pos_weight = get_pos_weight_val(sampled_target)

    rpn_cls_loss = F.binary_cross_entropy_with_logits(sampled_pred, sampled_target, pos_weight=pos_weight, reduction='mean')
    rpn_cls_loss = 0.5 * rpn_cls_loss

    rpn_reg_loss = F.smooth_l1_loss(pred_anch_locs, bb_y, beta=1, reduction='none')

    mask = torch.where(y > 0, 1, 0)
    mask = mask.permute(0, 2, 1)

    rpn_reg_loss = rpn_reg_loss * mask
    rpn_reg_loss = rpn_reg_loss.sum() / (rpn_reg_loss != 0).sum()

    rpn_loss = rpn_cls_loss + rpn_reg_loss

    return rpn_loss, fm, pred_cls_scores, pred_anch_locs