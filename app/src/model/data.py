import os
import torch 
from torch.utils.data import Dataset
import numpy as np
from torchio.transforms import RandomFlip, RandomAffine

class CropDataset(Dataset):
    def __init__(self, img_paths, label_paths, transform=None):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        x = np.load(self.img_paths[index])
        y = np.load(self.label_paths[index])
        
        fname = self.img_paths[index]
        
        x = torch.from_numpy(x['img']).float()
        labels = torch.from_numpy(y['labels']).float()
        locs = torch.from_numpy(y['locs']).float() 

        return fname, x, labels, locs