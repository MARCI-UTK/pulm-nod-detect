import torch 
import torch.nn as nn
import torch.nn.functional as F

from src.util.util import get_pos_weight_val, update_cm, rpn_to_roi

from src.model.feature_extractor import FeatureExtractor
from rpn import RPN

def crop_from_fm(fm: torch.Tensor, proposal_corners: torch.Tensor, scale: int):
    rv = []
    
    for i in range(len(fm)): 

        proposals = []
        for j in range(len(fm[0])):
            x1 = int(proposal_corners[i][j][0][0] / scale)
            y1 = int(proposal_corners[i][j][0][1] / scale)
            z1 = int(proposal_corners[i][j][0][2] / scale)

            x2 = int(proposal_corners[i][j][1][0] / scale)
            y2 = int(proposal_corners[i][j][1][1] / scale)
            z2 = int(proposal_corners[i][j][1][2] / scale)

            proposal_fm = fm[i, :, x1:x2, y1:y2, z1:z2]
            proposals.append(proposal_fm)

        rv.append(proposals)


    return torch.stack(rv)

class ROI(nn.Module): 
    def __init__(self): 
        super(ROI, self).__init__()

        self.flatten = torch.nn.Flatten()
        self.cls = torch.nn.Linear(1024, 1)
        self.reg = torch.nn.Linear(1024, 4)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(x.shape[0] * x.shape[1], -1).contiguous()
    
        cls = self.cls(x)
        reg = self.reg(x)

        cls = cls.view(orig_shape[0], orig_shape[1], -1).contiguous()
        reg = reg.view(orig_shape[0], orig_shape[1], -1).contiguous()
    
        return cls, reg 

class CropProposals(nn.Module): 
    def __init__(self): 
        super(CropProposals, self).__init__()

        self.pool = nn.AdaptiveMaxPool3d(output_size=2)

    def forward(self, fm: torch.Tensor, corners: torch.Tensor) -> torch.Tensor: 
        corners = corners / 4  

        LL = corners[:, :, 0, :]
        UR = corners[:, :, 1, :]

        LL = torch.clamp(LL, 0, 21).to(LL.device)
        UR = torch.where(UR - LL >= 2, UR, LL + 2).clamp(2, 23).to(UR.device)

        proposals = torch.empty(fm.shape[0], corners.shape[1], fm.shape[1], 2, 2, 2).to(fm.device)
        fm = fm.clone() 

        for i in range(corners.shape[0]): 
            for j in range(corners.shape[1]): 
                x1 = int(LL[i, j, 0].item())
                x2 = int(UR[i, j, 0].item())

                y1 = int(LL[i, j, 1].item())
                y2 = int(UR[i, j, 1].item())

                z1 = int(LL[i, j, 2].item())
                z2 = int(UR[i, j, 2].item())

                proposal = self.pool(fm[i, :, x1:x2, y1:y2, z1:z2])
                proposals[i, j, :, : , :, :] = proposal

        return proposals
    
def roi_iteration(y, bb_y, fm, anc_boxes, roi, cropper, 
                  optimizer, cls_scores, anc_locs, top_n): 

    # Make labels correct size 
    y = y.unsqueeze(1)

    y = y.to(f'cuda:{roi.device_ids[0]}')
    bb_y = bb_y.to(f'cuda:{roi.device_ids[0]}')

    corners, mask, indexs = rpn_to_roi(cls_scores=cls_scores, pred_locs=anc_locs, 
                                       anc_boxes=anc_boxes, nms_thresh=0.1, top_n=top_n)
    
    proposals = cropper(fm, corners)

    pred_cls_scores, pred_anch_locs = roi(proposals)

    top_n_y = []
    top_n_bb_y = []
    final_mask = []
    y = y.squeeze()
    pred_cls_scores = pred_cls_scores.squeeze()
    for i in range(len(y)): 
        y_i = y[i]
        b_i = bb_y[i] 
        m_i = mask[i]

        top_n_y.append(y_i[indexs[i]])
        top_n_bb_y.append(b_i[indexs[i]])
        final_mask.append(m_i[indexs[i]])

    top_n_y = torch.stack(top_n_y)
    top_n_bb_y = torch.stack(top_n_bb_y)
    final_mask = torch.stack(final_mask)

    #update_cm(top_n_y, pred_cls_scores, train_cm)

    final_mask = final_mask & (top_n_y != -1)

    pos_weight = get_pos_weight_val(top_n_y)

    roi_cls_loss = F.binary_cross_entropy_with_logits(pred_cls_scores, top_n_y, pos_weight=pos_weight, reduction='none')
    roi_cls_loss = roi_cls_loss.clone() * final_mask.float()
    roi_cls_loss = roi_cls_loss.sum() / (roi_cls_loss != 0).sum()
    roi_cls_loss = 0.5 * roi_cls_loss

    final_mask = final_mask & (top_n_y == 1)

    roi_reg_loss = F.smooth_l1_loss(pred_anch_locs, top_n_bb_y, reduction='none', beta=1)
    roi_reg_loss = roi_reg_loss.clone() * final_mask.unsqueeze(2).float()

    if (roi_reg_loss != 0).sum() == 0: 
        roi_reg_loss = 0
    else: 
        roi_reg_loss = roi_reg_loss.sum() / (roi_reg_loss != 0).sum()

    roi_loss = roi_reg_loss + roi_cls_loss   

    return roi_loss