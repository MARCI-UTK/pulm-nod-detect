import torch 
import torch.nn as nn 

class CQAM(nn.Module): 
    def __init__(self, in_channels: int, mid_channels: int, n_anchor: int): 
        super(CQAM, self).__init__()

        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.conv1 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3)
        self.bn1   = nn.BatchNorm3d(num_features=128)

        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.conv2 = nn.Conv3d()
        self.bn2   = nn.BatchNorm3d()

        self.pool3 = nn.MaxPool3d()
        self.conv3 = nn.Conv3d()
        self.bn3   = nn.BatchNorm3d()

        self.pool4 = nn.MaxPool3d()
        self.conv4 = nn.Conv3d()
        self.bn4   = nn.BatchNorm3d()
        
    def forward(self, x): 
        c1 = torch.permute(x, (1, 0, 2, 3))
        c2 = torch.permute(x, (2, 1, 0, 3))
        c3 = torch.permute(x, (3, 1, 2, 0))

        print(c1.shape, c2.shape, c3.shape)

        return 