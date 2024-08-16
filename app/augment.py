import os
import glob
import json 
import torch
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch.nn.functional

import monai

from src.dataModels.scan import CleanScan

data_path = '/data/marci/dlewis37/luna16'
scan_paths = os.path.join(data_path, 'processed_scan', '*.npy')
scans = glob.glob(scan_paths)

label_paths = os.path.join(data_path, 'processed_scan', '*.json')
labels = glob.glob(label_paths)

def world_to_vox(point: tuple, origin: tuple, spacing: tuple) -> tuple: 
    world_x, world_y, world_z, diameter = point
    origin_x, origin_y, origin_z        = origin
    spacing_x, spacing_y, spacing_z     = spacing

    voxel_x = (world_x - origin_x) / spacing_x
    voxel_y = (world_y - origin_y) / spacing_y
    voxel_z = (world_z - origin_z) / spacing_z

    voxel_diameter = diameter / spacing_x

    voxel_point = [voxel_x, voxel_y, voxel_z, voxel_diameter]

    return voxel_point

def make_plt_rect(xyzd, color): 
    x = xyzd[0] - xyzd[3] // 2
    y = xyzd[1] - xyzd[3] // 2
    d = xyzd[3]

    rect = Rectangle(xy=(x, y), width=d, height=d, fill=False, color=color)
    return rect 

def scanToCropCoordinate(center, nodule_voxel_location): 
    vox_x, vox_y, vox_z, diameter = nodule_voxel_location
    x_c, y_c, z_c = center

    crop_loc_x, crop_loc_y, crop_loc_z = (vox_x - x_c, vox_y - y_c, vox_z - z_c)
    
    return (crop_loc_x, crop_loc_y, crop_loc_z, diameter)

def plot_scan(img, lbl, save=False): 
    rect = [make_plt_rect(x, 'r') for x in lbl]
    
    if len(lbl) == 0: 
        slice_idx = len(img) // 2

    for i in range(len(lbl)): 
        slice_idx = int(lbl[i][2])

        plt.imshow(img[slice_idx], cmap=plt.bone())
        plt.gca().add_patch(rect[i])

        if save: 
            plt.savefig(save)

        plt.show()
        plt.cla()

def flip(img, lbl, axis): 
    #flipper = monai.transforms.Flip(spatial_axis=axis)
    #rv = flipper(img)
    rv = np.flip(img, axis=axis)

    """
    The image in the numpy array is in Z, Y, X format thus the flip axis 
    will be 1 or 2. 
    The labels are in X, Y, Z, D format. Thus to change the label for a flip 
    of the 2nd (X) axis in the image, it will be the 0th index in the label. 
    """
    lbl_axis = 2 - axis

    new_lbl = np.copy(lbl)
    new_lbl[0][lbl_axis] = img.shape[axis] - lbl[0][lbl_axis] - 1

    return rv, new_lbl

def scale(img, lbl, factor): 
    zoomer = monai.transforms.Zoom(zoom=factor, padding_mode='empty')
    rv = zoomer(img)

    pcts = [x / 96 for x in lbl]
    new_lbl = [p * (96 * factor) for p in pcts]

    return rv, new_lbl

def add_noise(img): 
    noise = np.random.normal(0, 0.01, size=img.shape)
    rv = img + noise

    return rv 

def shift_crop(img, nod_loc, s_bound): 
    check = True
    center = []

    if nod_loc[0] < 48 or nod_loc[1] < 48 or nod_loc[2] < 48: 
        return None
    elif nod_loc[0] > img.shape[2] - 48 or nod_loc[1] > img.shape[1] - 48 or nod_loc[2] > img.shape[0] - 48: 
        return None

    while check: 
        if s_bound != 0: 
            center = [np.random.randint(-1 * s_bound, s_bound) + nod_loc[i] for i in range(3)]
        else: 
            center = nod_loc

        center = list(map(int, center))
        
        origin = [x - 48 for x in center]

        x_c = center[0]
        y_c = center[1]
        z_c = center[2]

        if x_c < 48 or y_c < 48 or z_c < 48: 
            continue 
        elif x_c > img.shape[2] - 48 or y_c > img.shape[1] - 48 or z_c > img.shape[0] - 48: 
            continue  

        check = False

        nodule_loc = [nod_loc[i] - origin[i] for i in range(3)]
        d = [nod_loc[3]]
        bbox = nodule_loc + d

        crop = img[origin[2]:origin[2] + 96, origin[1]:origin[1] + 96, origin[0]:origin[0] + 96]

    return crop, bbox

# 1185 total nodules (multiple per scan) !!

for s in scans: 
    scan = CleanScan(s)
    
    img = scan.img
    lbl = scan.annotations

    img = (img - np.min(img)) / (np.max(img) - np.min(img)) 

    if len(lbl) != 0: 
        lbl = [world_to_vox(x, scan.origin, scan.spacing) for x in lbl] 

        img = add_noise(img)
        img, lbl = flip(img, lbl, 1)

        crop_rv = shift_crop(img, lbl[0], 25)

        if crop_rv == None: 
            continue

        crop, c_lbl = crop_rv

        plt.imshow(crop[int(c_lbl[2])], cmap=plt.bone())
        plt.gca().add_patch(make_plt_rect(c_lbl, 'r'))
        plt.title("Augmented Crop")
        plt.savefig('augmented_crop.png')
        plt.show()
        plt.cla()

        """
        scaled, s_lbl = scale(crop, c_lbl, 0.75)

        plt.imshow(scaled[int(s_lbl[2])], cmap=plt.bone())
        rect = make_plt_rect(s_lbl, 'r')
        plt.gca().add_patch(rect)
        plt.title('% technique')
        plt.show()
        plt.cla()
        """

        """
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(18, 9))

        ax1.imshow(img[int(lbl[0][2])], cmap=plt.bone())
        ax1.set_title("Original")
        ax1.add_patch(make_plt_rect(lbl[0], 'r'))

        ax2.imshow(noisy[int(lbl[0][2])], cmap=plt.bone())
        ax2.set_title("Noise")
        ax2.add_patch(make_plt_rect(lbl[0], 'r'))

        ax3.imshow(flipped[int(f_lbl[0][2])], cmap=plt.bone())
        ax3.set_title("Flip")
        ax3.add_patch(make_plt_rect(f_lbl[0], 'r'))

        plt.show()
        plt.cla()
        """

        out_path = f'imgs/{scan.scanId}.png'
        #plot_scan(img, lbl, save=out_path)

"""
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(12, 6))
ax1.imshow(img[img.shape[0] // 2], cmap=plt.bone())
ax1.set_title("Original")
ax2.imshow(zoomed[zoomed.shape[0] // 2], cmap=plt.bone())
ax2.set_title("Zoomed")
plt.show()
plt.cla()
"""