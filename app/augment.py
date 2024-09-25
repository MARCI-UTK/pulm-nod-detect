import concurrent.futures
import os
import glob
from tqdm import tqdm
import numpy as np 
from scipy.ndimage import zoom
import concurrent

from src.dataModels.scan import CleanScan
from src.util.util import powerset

def get_neg_crop(img): 
    randX = np.random.randint(0, img.shape[2] - 96)
    randY = np.random.randint(0, img.shape[1] - 96)
    randZ = np.random.randint(0, img.shape[0] - 96)

    crop = img[randZ:randZ + 96, randY:randY + 96, randX:randX + 96]

    return crop

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

def scanToCropCoordinate(center, nodule_voxel_location): 
    vox_x, vox_y, vox_z, diameter = nodule_voxel_location
    x_c, y_c, z_c = center

    crop_loc_x, crop_loc_y, crop_loc_z = (vox_x - x_c, vox_y - y_c, vox_z - z_c)
    
    return (crop_loc_x, crop_loc_y, crop_loc_z, diameter)

def flip(img, lbl): 
    axis = np.random.random()
    axis = 1 if axis < 0.5 else 2

    rv = np.flip(img, axis=axis)

    """
    The image in the numpy array is in Z, Y, X format thus the flip axis 
    will be 1 or 2. 
    The labels are in X, Y, Z, D format. Thus to change the label for a flip 
    of the 2nd (X) axis in the image, it will be the 0th index in the label. 
    """
    lbl_axis = 2 - axis

    new_lbl = np.copy(lbl)
    new_lbl[lbl_axis] = img.shape[axis] - lbl[lbl_axis] - 1

    return rv, new_lbl

def scale(img, lbl): 
    if np.random.random() < 0.5: 
        factor = np.random.uniform(0.75, 0.85)
    else: 
        factor = np.random.uniform(1.15, 1.25)

    orig_z, orig_y, orig_x = img.shape

    new_img = zoom(img, (1, factor, factor))

    x_scale = new_img.shape[2] / orig_x
    y_scale = new_img.shape[1] / orig_y
    z_scale = new_img.shape[0] / orig_z

    scales = [x_scale, y_scale, z_scale]

    new_lbl = [lbl[i] * scales[i] for i in range(3)]
    new_lbl.append(lbl[3] * factor) 

    return new_img, new_lbl

def add_noise(img, lbl): 
    dark_mask = (img == 0)
    light_mask = (img == 1)
    
    noise = np.random.normal(0, 0.05, size=img.shape)
    rv = img + noise

    # Restore black outer area and cap values at 1
    rv[dark_mask] = 0
    rv[light_mask] = 1

    return rv, lbl

def check_invalid_nod(img, lbl):
    if lbl[0] < 48 or lbl[1] < 48 or lbl[2] < 48: 
        return True
    elif lbl[0] > img.shape[2] - 48 or lbl[1] > img.shape[1] - 48 or lbl[2] > img.shape[0] - 48: 
        return True

    return False

def shift_crop(img, nod_loc, s_bound): 
    check = True
    center = []

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

def augment(scan): 
    count = 0

    ops = [0, 1, 2]
    ops = list(powerset(ops))
    ops = [list(x) for x in ops]

    scan = CleanScan(scan)

    img = scan.img
    lbl = scan.annotations

    img = (img - np.min(img)) / (np.max(img) - np.min(img)) 

    if len(lbl) != 0: 
        lbl = [world_to_vox(x, scan.origin, scan.spacing) for x in lbl]

        for l in lbl: 

            invalid = check_invalid_nod(img, l)

            if invalid: 
                continue

            aug_imgs = []
            aug_lbls = []

            keep = np.random.permutation(len(ops))
            ops = [ops[i] for i in keep]
            
            for op_set in ops: 
                aug_i, aug_l = img, l

                if len(op_set) == 0:
                    aug_i, aug_l = shift_crop(aug_i, aug_l, 24)

                    aug_imgs.append(aug_i)
                    aug_lbls.append(aug_l)

                    continue

                for op in op_set: 

                    if op == 0: 
                        aug_i, aug_l = add_noise(aug_i, aug_l)
                    if op == 1: 
                        aug_i, aug_l = flip(aug_i, aug_l)
                    if op == 2: 
                        aug_i, aug_l = scale(aug_i, aug_l)

                aug_i, aug_l = shift_crop(aug_i, aug_l, 24)
                    
                aug_imgs.append(aug_i)
                aug_lbls.append(aug_l)

            for i in range(len(aug_imgs)): 

                outpath = os.path.join(data_path, 'crops', f'{scan.scanId}_{str(count)}.npz')
                np.savez_compressed(file=outpath,
                                    img=[aug_imgs[i],],
                                    label=1, 
                                    bbox=aug_lbls[i],)

                count += 1

        else: 
            l = [0, 0, 0, 0]

            keep = np.random.permutation(len(ops))
            ops = [ops[i] for i in keep]
            ops = ops[0]

            aug_i = img

            for op in ops: 

                if op == 0: 
                    aug_i, aug_l = add_noise(aug_i, l)
                if op == 1: 
                    aug_i, aug_l = flip(aug_i, l)
                if op == 2: 
                    aug_i, aug_l = scale(aug_i, l)

            crop = get_neg_crop(aug_i) 

            outpath = os.path.join(data_path, 'crops', f'{scan.scanId}_{str(count)}.npz')

            np.savez_compressed(file=outpath,
                                img=[crop],
                                label=0, 
                                bbox=[0, 0, 0, 0],)

            count += 1
    
# 1185 total nodules (multiple per scan) !!

"""
ops = [0, 1, 2]
ops = list(powerset(ops))
ops = [list(x) for x in ops]
"""

if __name__ == "__main__":
    np.random.seed(27)

    data_path = '/data/marci/luna16'
    scan_paths = os.path.join(data_path, 'processed_scan', '*.npy')
    scans = glob.glob(scan_paths)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=32)

    with tqdm(scans) as pbar: 
        futures = [pool.submit(augment, s) for s in scans]

        for future in concurrent.futures.as_completed(futures): 
            pbar.update(1)
        
    pool.shutdown(wait=True)
