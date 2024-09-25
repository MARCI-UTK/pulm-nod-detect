import torch.nn as nn
import torch

class Bottleneck3d(nn.Module):

    def __init__(self, in_channels, intermediate_channels, stride):
        
        """
        Creates a Bottleneck with conv 1x1->3x3->1x1 layers.
        
        Note:
          1. Addition of feature maps occur at just before the final ReLU with the input feature maps
          2. if input size is different from output, select projected mapping or else identity mapping.
          3. if is_Bottleneck=False (3x3->3x3) are used else (1x1->3x3->1x1). Bottleneck is required for resnet-50/101/152
        Args:
            in_channels (int) : input channels to the Bottleneck
            intermediate_channels (int) : number of channels to 3x3 conv 
            expansion (int) : factor by which the input #channels are increased
            stride (int) : stride applied in the 3x3 conv. 2 for first Bottleneck of the block and 1 for remaining

        Attributes:
            Layer consisting of conv->batchnorm->relu

        """

        super().__init__()

        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels

        if self.in_channels == self.intermediate_channels:
            self.identity = True
        else:
            self.identity = False
            projection_layer = []
            projection_layer.append(nn.Conv3d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            projection_layer.append(nn.BatchNorm3d(self.intermediate_channels))

            # Only conv->BN and no ReLU
            # projection_layer.append(nn.ReLU())

            self.projection = nn.Sequential(*projection_layer)

        # commonly used relu
        self.relu = nn.ReLU()

        # bottleneck
        # 1x1
        self.conv1_1x1 = nn.Conv3d(in_channels=self.in_channels, out_channels=self.intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False )
        self.batchnorm1 = nn.BatchNorm3d(self.intermediate_channels)
        
        # 3x3
        self.conv2_3x3 = nn.Conv3d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False )
        self.batchnorm2 = nn.BatchNorm3d(self.intermediate_channels)
        
        # 1x1
        self.conv3_1x1 = nn.Conv3d(in_channels=self.intermediate_channels, out_channels=self.intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False )
        self.batchnorm3 = nn.BatchNorm3d(self.intermediate_channels) 

    def forward(self,x):
        # input stored to be added before the final relu
        skip_x = x

        # conv1x1->BN->relu
        x = self.conv1_1x1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        # conv3x3->BN->relu
        x = self.conv2_3x3(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        
        # conv1x1->BN
        x = self.conv3_1x1(x)
        x = self.batchnorm3(x)
        
        # Add skip connection
        if (x.shape == skip_x.shape):
            x += skip_x
        else:
            x += self.projection(skip_x)
        
        # final relu
        x = self.relu(x)

        return x
    
def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out