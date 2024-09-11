import matplotlib
matplotlib.use('Agg')

from .util import voxel_to_world

import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


from .util import worldToVoxel, get_iou
from ..dataModels.crop import Crop
from ..dataModels.scan import CleanScan

def scanToCropCoordinate(center, nodule_voxel_location): 
    vox_x, vox_y, vox_z, diameter = nodule_voxel_location
    x_c, y_c, z_c = center

    crop_loc_x, crop_loc_y, crop_loc_z = (vox_x - x_c, vox_y - y_c, vox_z - z_c)
    
    return (crop_loc_x, crop_loc_y, crop_loc_z, diameter)

def get_neg_crop(img): 
    randX = np.random.randint(0, img.shape[2] - 96)
    randY = np.random.randint(0, img.shape[1] - 96)
    randZ = np.random.randint(0, img.shape[0] - 96)

    crop = img[randZ:randZ + 96, randY:randY + 96, randX:randX + 96]

    return crop

def generateCrops(dataPath: str): 
    
    # Total statistics 
    total_crops = 0
    pos = 0
    neg = 0 

    for npyFile in glob.glob(os.path.join(dataPath, 'processed_scan', '*.npy')):   
        scan = CleanScan(npyPath=npyFile) 

        if scan.img.shape[0] < 97: 
            continue
 
        img = scan.img
        img = (img - np.min(img)) / (np.max(img) - np.min(img)) 
        
        if len(scan.annotations) != 0: 
            for i in scan.annotations: 
                nodule_voxel_location  = worldToVoxel(world_point=i, world_origin=scan.origin, 
                                                      spacing=scan.spacing)
                
                nodule_voxel_location = list(nodule_voxel_location)

                center = [np.random.randint(-10, 11) + nodule_voxel_location[i] for i in range(3)]
                origin = [x - 48 for x in center]

                x_c = center[0]
                y_c = center[1]
                z_c = center[2]

                if x_c < 48 or y_c < 48 or z_c < 48: 
                    continue 
                elif x_c > img.shape[2] - 48 or y_c > img.shape[1] - 48 or z_c > img.shape[0] - 48: 
                    continue  

                nodule_loc = [nodule_voxel_location[i] - origin[i] for i in range(3)]
                d = [nodule_voxel_location[3]]
                bbox = nodule_loc + d
 
                crop = img[origin[2]:origin[2] + 96, origin[1]:origin[1] + 96, origin[0]:origin[0] + 96]

                outpath = os.path.join(dataPath, 'dataset', f'{scan.scanId}_{str(total_crops)}.npz')
                np.savez_compressed(file=outpath,
                                    img=[crop,],
                                    label=1, 
                                    bbox=bbox,)
                
                print(f'wrote to {outpath}')

                pos += 1
                total_crops += 1 

        else: 
            crop = get_neg_crop(img)

            outpath = os.path.join(dataPath, 'dataset', f'{scan.scanId}_{str(total_crops)}.npz')
            np.savez_compressed(file=outpath,
                                img=[crop,],
                                label=0, 
                                bbox=[0, 0, 0, 0],)
            
            print(f'wrote to {outpath}')
            
            neg += 1
            total_crops += 1

    print(f'total crops: {total_crops}. positive: {pos}. negative: {neg}')
    