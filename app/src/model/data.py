import os
import torch 
from torch.utils.data import Dataset
import numpy as np

class ROIDataset(Dataset):
    def __init__(self, paths, transform=None):
        
        return 
            

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        data = np.load(self.paths[index])
        
        fname = self.paths[index]
        
        x = torch.from_numpy(data['proposal']).float()
        anc_box = torch.from_numpy(data['anc_box']).float()
        gt_delta = torch.from_numpy(data['gt_delta']).float()
        y = torch.from_numpy(data['y']).float()

        if self.transform:
            x = self.transform(x)

        return fname, x, y, gt_delta, anc_box

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

        if self.transform:
            x = self.transform(x)

        return fname, x, labels, locs