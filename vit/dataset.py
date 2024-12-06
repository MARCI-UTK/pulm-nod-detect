import torch 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.ndimage import zoom

class CropDataset(Dataset):
    def __init__(self, img_paths, label_paths, transform=None):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        data = np.load(self.img_paths[index])
        x = torch.from_numpy(data['img']).float()
        y = data['label'].astype(float)
       
        
        fname = self.img_paths[index]
        
        #labels = torch.from_numpy(y['labels']).float()
        #locs = torch.from_numpy(y['locs']).float() 

        return fname, x, y
    
class ScanDataset(Dataset):
    def __init__(self, img_paths, label_path, transform=None):
        self.img_paths = img_paths
        self.labels = pd.read_csv(label_path)
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = np.load(self.img_paths[index], allow_pickle=True)
        
        img = torch.from_numpy(img).float().cuda()

        img = img.unsqueeze(0).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, (128, 128, 128), mode='trilinear', align_corners=False)
        img = img.squeeze(0)

        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img)) 

        id = self.img_paths[index].split('/')[-1][0:-4]

        label = self.labels[self.labels['seriesuid'] == id]
        label = 1. if len(label) != 0 else 0.     
        
        fname = self.img_paths[index]
        
        #labels = torch.from_numpy(y['labels']).float()
        #locs = torch.from_numpy(y['locs']).float() 

        return fname, img, label