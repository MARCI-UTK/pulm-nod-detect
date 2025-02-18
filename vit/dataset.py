import os
import json
import util
import torch
import scipy
import einops 
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class ScanDataset(Dataset):
    def __init__(self, root_path, transform=None):
        img_paths, metadata_paths, annotations_path = self.make_paths(root_path=root_path)

        # Paths of scans 
        self.img_paths = img_paths

        # Paths of scan metadata (origin and spacing of scan)
        self.metadata_paths = metadata_paths

        # Annotations for scans 
        self.annotations = pd.read_csv(annotations_path)

        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # Load pre-processed CT scan
        img = np.load(self.img_paths[index], allow_pickle=True)
        img = img['img']
        
        # Convert to float tensor 
        img = torch.from_numpy(img).float()

        # Used to resize bounding box for nodule to match new image size 
        orig_shape = list(img.shape)

        tmp = orig_shape[0]
        orig_shape[0] = orig_shape[2]
        orig_shape[2] = tmp

        # Introduce the B and C dimensions in order to use torch's interpolate function 
        # For a 3D interpolation, function expects input of [B, C, H, W, D]
        img = img.unsqueeze(0).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, (128, 128, 128), mode='trilinear', align_corners=False)

        # Remove the added B dimension as it will be added back during dataloader creation
        img = img.squeeze(0)

        # Normalize voxel values to 0-1 range 
        img = (img - torch.min(img)) / (torch.max(img) - torch.min(img)) 

        # Get ID of scan and annotations associated with that ID 
        id = self.img_paths[index].split('/')[-1][0:-4]
        annotations = self.annotations[self.annotations['seriesuid'] == id]

        # Label denotes if a scan contains a nodule 
        # 1 if yes, 0 if not 
        label = 1. if len(annotations) != 0 else 0.

        # Get the bounding box for the nodule from the annotations CSV
        locs = []
        if label == 1: 
            for _, row in annotations.iterrows(): 
                locs = [row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm']]
                break

            # Read scan metadata (origin and spacing)
            with open(self.metadata_paths[index], 'r') as f: 
                metadata = json.load(f)
                f.close()

            origin = metadata['origin']
            spacing = metadata['spacing']

            # Convert nodule location from world coordinates to voxel coordinates 
            locs = util.world_2_voxel(locs, origin, spacing)
            locs = util.adjust_bbox(locs, orig_shape, 128)

            # Double the diameter of what will be selected from scan to make nodule mask
            # Ensures nodule is fully enclosed in section
            locs[3] = locs[3] * 2

            # Negative voxel coordiantes cause problems
            locs = np.abs(locs)

            # Convert to corners to crop nodule from scan
            c1, c2 = util.xyzd_2_2corners(locs)

            x1, y1, z1 = c1
            x2, y2, z2 = c2

            # Select nodule from scan and create mask of nodule 
            nodule = img[:, int(z1):int(z2) + 1, int(y1):int(y2) + 1, int(x1):int(x2) + 1]
            nodule = (nodule - torch.min(nodule)) / (torch.max(nodule) - torch.min(nodule)) 

            labeled_mask, num_features = scipy.ndimage.label(nodule > 0.7)
            sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
            biggest_component = np.argmax(sizes) + 1
            biggest_component_mask = (labeled_mask == biggest_component)

            # Insert nodule mask into scan size mask
            mask = np.zeros_like(img)
            mask[:, int(z1):int(z2) + 1, int(y1):int(y2) + 1, int(x1):int(x2) + 1] = biggest_component_mask

        # If scan doesn't have a nodule, set bounding box to 0's
        else: 
            locs = [0., 0., 0., 0.]
            mask = np.zeros_like(img)

        locs = torch.tensor(locs).float()
        mask = torch.tensor(mask).float()

        return img, mask, locs, label
    
    def make_paths(self, root_path): 
        img_metadata_paths = [os.path.join(root_path, 'processed_scan', f) for f in os.listdir(os.path.join(root_path, 'processed_scan'))]

        img_paths = [p for p in img_metadata_paths if p.endswith('.npz')]
        metadata_paths = [p for p in img_metadata_paths if p.endswith('.json')]

        annotations_path = os.path.join(root_path, 'csv', 'annotations.csv')

        return img_paths, metadata_paths, annotations_path
    
class MSD_Dataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        return 