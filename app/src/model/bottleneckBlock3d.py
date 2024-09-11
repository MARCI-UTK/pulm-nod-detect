import torch.nn as nn
import torch
from hflayers import HopfieldLayer

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
        
        # i.e. if dim(x) == dim(F) => Identity function
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

        self.hopfield = HopfieldLayer(input_size=self.intermediate_channels,
                                      lookup_weights_as_separated=True,
                                      lookup_targets_as_trainable=False,
                                      normalize_stored_pattern_affine=True,)

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
        
        # identity or projected mapping
        if (x.shape == skip_x.shape):
            x += skip_x
        else:
            x += self.projection(skip_x)
        
        #x = x.view(32, -1, self.intermediate_channels).contiguous()

        orig_shape = x.shape

        x = x.view(orig_shape[0], -1, self.intermediate_channels).contiguous()
        x = self.hopfield(x)
        x = x.reshape(orig_shape).contiguous()

        # final relu
        x = self.relu(x)

        return x