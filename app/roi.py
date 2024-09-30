import torch 
import torch.nn as nn

from src.util.util import apply_bb_deltas, xyzd_2_2corners

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

        """
        for j in range(corners.shape[1]): 
            x1 = LL[:, j, 0].int()
            x2 = UR[:, j, 0].int()

            y1 = LL[:, j, 1].int()
            y2 = UR[:, j, 1].int()

            z1 = LL[:, j, 2].int()
            z2 = UR[:, j, 2].int()

            proposal = self.pool(fm[:, :, x1:x2, y1:y2, z1:z2])
            proposals[:, j, :, : , :, :] = proposal
        """

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