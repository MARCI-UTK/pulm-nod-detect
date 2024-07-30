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
        x = self.flatten(x)
    
        cls = self.cls(x)
        reg = self.reg(x)
    
        return cls, reg 

class CropProposals(nn.Module): 
    def __init__(self): 
        super(CropProposals, self).__init__()

        self.pool = nn.AdaptiveMaxPool3d(output_size=2)

    def forward(self, fm, corners, scale): 
        rv = torch.zeros((len(fm), len(corners[0]), 128, 2, 2, 2))
    
        for i in range(len(fm)): 
            for j in range(len(corners[0])):

                p1 = [int(corners[i][j][0][x] / scale) for x in range(3)]
                p1 = [p1[x] if p1[x] > 0 else 0 for x in range(3)]
                p1 = [p1[x] if p1[x] < 21 else 21 for x in range(3)]
                
                p2 = [int(corners[i][j][1][x] / scale) for x in range(3)]
                p2 = [p2[x] if (p2[x] - p1[x]) >= 2 else p1[x] + 2 for x in range(3)]

                x1, y1, z1 = p1
                x2, y2, z2 = p2

                #print(f'p1: {x1, y1, z1}. p2: {x2, y2, z2}')

                proposal_fm = fm[i, :, x1:x2, y1:y2, z1:z2]
                proposal_fm = self.pool(proposal_fm)

                rv[i][j] = proposal_fm

        return rv