import numpy as np 
import itertools
import os
from src.util.util import get_iou, xyzd_2_2corners, corners_2_xyzd
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
import torch.nn as nn 
import torch

from src.model.feature_extractor import FeatureExtractor

# Output of Nature paper before RPN is 24x24x24x128

class RPN(nn.Module): 
    def __init__(self, in_channels: int, mid_channels: int, n_anchor: int): 
        super(RPN, self).__init__()

        self.fe = FeatureExtractor()

        self.conv1    = nn.Conv3d(in_channels=in_channels, out_channels=512, 
                                  kernel_size=3, stride=1, padding=1)
        
        self.conv_cls = nn.Conv3d(in_channels=512, out_channels=n_anchor, kernel_size=1, stride=1)
        self.conv_reg = nn.Conv3d(in_channels=512, out_channels=n_anchor * 4, kernel_size=1, stride=1)

        # Initialize weights and biases as described in Faster-RCNN paper
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()

        self.conv_cls.weight.data.normal_(0, 0.01)
        self.conv_cls.bias.data.zero_()

        self.conv_reg.weight.data.normal_(0, 0.01)
        self.conv_reg.bias.data.zero_()
        """
        
    
    def forward(self, x): 
        fm = self.fe(x)

        x = fm
        x = self.conv1(x)

        pred_anc_locs = self.conv_reg(x)
        pred_cls_scores = self.conv_cls(x)

        pred_anc_locs = pred_anc_locs.permute(0, 2, 3, 4, 1).contiguous()
        pred_anc_locs = pred_anc_locs.view(pred_anc_locs.shape[0], 24 * 24 * 24 * 3, 4)

        pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 4, 1).contiguous()
        pred_cls_scores = pred_cls_scores.view(pred_cls_scores.shape[0], 1, 24 * 24 * 24 * 3) 

        return pred_anc_locs, pred_cls_scores, fm

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