import torch 
import torch.nn as nn

class ResBlock(nn.Module): 
    def __init__(self, in_channels, out_channels): 
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm3d(num_features=in_channels)

        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm3d(num_features=in_channels)
        
        self.relu = nn.ReLU()

    def forward(self, x): 
        skip_x = x 

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        print(x.shape)
        print(skip_x.shape)

        x += skip_x
        x = self.relu(x)

        return x 

