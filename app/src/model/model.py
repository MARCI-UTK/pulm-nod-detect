import math
import torch 
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class Bottle2Neck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super().__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv3d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width*scale)

        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool3d(kernel_size=3, stride = stride, padding=1)

        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv3d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm3d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv3d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]

          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))

          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class TestModel(nn.Module): 
    def __init__(self):
        super().__init__() 

        self.conv1 = nn.Conv3d(1, 1, 3)

        self.bottle2neck3d = Bottle2Neck3d(inplanes=1, planes=3)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(778688, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
       
      x = self.conv1(x)

      x = self.bottle2neck3d(x)
      
      x = self.flatten(x)
      x = F.relu(self.fc1(x))
      x = torch.sigmoid(self.fc2(x))

      return x
    