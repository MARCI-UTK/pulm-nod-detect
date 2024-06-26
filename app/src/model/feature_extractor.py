import torch 
import torch.nn as nn
from src.model.bottleneckBlock3d import Bottleneck3d

class FeatureExtractor(nn.Module): 
    def __init__(self): 
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)

        self.mp1 = nn.MaxPool3d(2, 2)

        self.bottle1 = Bottleneck3d(in_channels=24, intermediate_channels=28, stride=1)
        self.bottle2 = Bottleneck3d(in_channels=28, intermediate_channels=32, stride=1)

        self.mp2 = nn.MaxPool3d(2, 2)

        self.bottle3 = Bottleneck3d(in_channels=32, intermediate_channels=48, stride=1)
        self.bottle4 = Bottleneck3d(in_channels=48, intermediate_channels=64, stride=1)

        self.mp3 = nn.MaxPool3d(2, 2)

        self.bottle5 = Bottleneck3d(in_channels=64, intermediate_channels=64, stride=1)
        self.bottle6 = Bottleneck3d(in_channels=64, intermediate_channels=64, stride=1)
        self.bottle7 = Bottleneck3d(in_channels=64, intermediate_channels=64, stride=1)

        self.mp4 = nn.MaxPool3d(2, 2)

        self.bottle8  = Bottleneck3d(in_channels=64, intermediate_channels=64, stride=1)
        self.bottle9  = Bottleneck3d(in_channels=64, intermediate_channels=64, stride=1)
        self.bottle10 = Bottleneck3d(in_channels=64, intermediate_channels=64, stride=1)

        self.up1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=2, stride=2)

        self.bottle11 = Bottleneck3d(in_channels=64, intermediate_channels=64, stride=1)
        self.bottle12 = Bottleneck3d(in_channels=64, intermediate_channels=64, stride=1)
        self.bottle13 = Bottleneck3d(in_channels=64, intermediate_channels=64, stride=1)

        self.up2 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=2, stride=2)

        # This could be (64 -> 96) -> (96 -> 128) -> (128 -> 128)
        # OR could be (64 -> 64) -> (64 -> 96) -> (96 -> 128)
        # OR could be (64 -> 96) -> (96 -> 96) -> (96 -> 128)
        self.bottle14 = Bottleneck3d(in_channels=64, intermediate_channels=96, stride=1)
        self.bottle15 = Bottleneck3d(in_channels=96, intermediate_channels=128, stride=1)
        self.bottle16 = Bottleneck3d(in_channels=128, intermediate_channels=128, stride=1) 

    def forward(self, x): 
        # Input is 1x96x96x96 here 
        x = self.conv1(x)
        x = self.conv2(x)

        # Input is 24x96x96x96 here
        x = self.mp1(x)

        # Input is 24x48x48x48 here
        x = self.bottle1(x)
        x = self.bottle2(x)

        # Input is 32x48x48x48 here 
        x = self.mp2(x)

        # Input is 32x24x24x24 here 
        x = self.bottle3(x)
        x = self.bottle4(x)
        
        # Input is 64x24x24x24 here
        res1 = x
        x = self.mp3(x)

        # Input is 64x12x12x12 here 
        x = self.bottle5(x)
        x = self.bottle6(x)
        x = self.bottle7(x)

        # Input is 64x12x12x12 here
        res2 = x 
        x = self.mp4(x)

        # Input is 64x6x6x6 here 
        x = self.bottle8(x)
        x = self.bottle9(x)
        x = self.bottle10(x)

        # Input is 64x6x6x6 here 
        x = self.up1(x)

        # Input is 64x12x12x12 here 
        x += res2

        x = self.bottle11(x)
        x = self.bottle12(x)
        x = self.bottle13(x)

        # Input is 64x12x12x12 here 
        x = self.up2(x)

        # Input is 64x24x24x24 here 
        x += res1

        x = self.bottle14(x)
        x = self.bottle15(x)
        x = self.bottle16(x)

        # x at this point is 128x24x24x24

        return x 