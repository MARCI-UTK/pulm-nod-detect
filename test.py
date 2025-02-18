import os
import json 
import glob
import numpy as np
import pandas as pd  
import SimpleITK as sitk
import scipy.ndimage

# Extract scan ID from path to scan 
def get_scan_id(path: str) -> str: 
    return path.split('/')[-1][:-4]

# Pad scan to be a cube by adding 0 slices to front and back of z-axis 
def pad_img(img: np.ndarray) -> np.ndarray:
    # Determine if padding needs to be on z or x/y axes
    diff = img.shape[2] - img.shape[0] 

    if diff == 0: 
        return img
    # Means z > x/y so we need to pad on x and y axis 
    elif diff < 0: 
        if diff % 2 == 0: 
            pad_amt = (img.shape[0] - img.shape[2]) // 2 
            img = np.pad(img, ((0, 0), (pad_amt, pad_amt), (pad_amt, pad_amt)), 'constant', constant_values=0)
        else: 
            pad_amt_1 = (img.shape[0] - img.shape[2]) // 2 
            pad_amt_2 = pad_amt_1 + 1
            img = np.pad(img, ((0, 0), (pad_amt_1, pad_amt_2), (pad_amt_1, pad_amt_2)), 'constant', constant_values=0)
        return img 
    # Means z < x/y so we need to pad on z axis  
    else: 
        if diff % 2 == 0: 
            pad_amt = (img.shape[2] - img.shape[0]) // 2 
            img = np.pad(img, ((pad_amt, pad_amt), (0, 0), (0, 0)), 'constant', constant_values=0)
        else: 
            pad_amt_1 = (img.shape[2] - img.shape[0]) // 2 
            pad_amt_2 = pad_amt_1 + 1
            img = np.pad(img, ((pad_amt_1, pad_amt_2), (0, 0), (0, 0)), 'constant', constant_values=0)
        return img

def world_2_voxel(world_point: tuple, world_origin: tuple, spacing: tuple) -> tuple: 
    world_x, world_y, world_z, diameter = world_point
    origin_x, origin_y, origin_z        = world_origin
    spacing_x, spacing_y, spacing_z     = spacing

    voxel_x = (world_x - origin_x) // spacing_x
    voxel_y = (world_y - origin_y) // spacing_y
    voxel_z = (world_z - origin_z) // spacing_z

    voxel_diameter = diameter / spacing_x

    voxel_point = (int(voxel_x), int(voxel_y), int(voxel_z), int(voxel_diameter))

    return voxel_point

data_path = '/data/marci/luna16/' 

# Dataset annotations (.csv)
annotations_path = os.path.join(data_path, 'csv', 'annotations.csv')
annotations = pd.read_csv(annotations_path)
    
img_paths = glob.glob(os.path.join(data_path, f'processed_scan', '*.npz')) 
json_paths = glob.glob(os.path.join(data_path, f'processed_scan', '*.json'))

l = len(img_paths)
for i in range(l):
    scan_id = get_scan_id(img_paths[i])
    scan_annotations = annotations[annotations['seriesuid'] == scan_id]

    if scan_annotations.shape[0] > 1: 
        print(scan_id)

    