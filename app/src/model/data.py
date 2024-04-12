import os
import torch 
from torch.utils.data import Dataset
import numpy as np

class CropDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        x_y = np.load(self.paths[index])

        x = torch.from_numpy(x_y['img']).float()
        y = torch.from_numpy(x_y['label']).float()

        if self.transform:
            x = self.transform(x)

        return x, y