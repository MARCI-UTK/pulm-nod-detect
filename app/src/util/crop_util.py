import os 
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .util import worldToVoxel
from ..dataModels.crop import Crop
from ..dataModels.scan import CleanScan


def cropCube(scan: np.array, numCubes: int) -> list: 
    crops = []

    for _ in range(numCubes):
        randX = np.random.randint(0, scan.shape[2] - 96)
        randY = np.random.randint(0, scan.shape[1] - 96)
        randZ = np.random.randint(0, scan.shape[0] - 96)

        crop = scan[randZ:randZ + 96, randY:randY + 96, randX:randX + 96]

        crops.append((crop, (randX, randY, randZ)))

    return crops

def scanToCropNoduleLocation(anchor_point, nodule_voxel_location, spacing): 
    vox_x, vox_y, vox_z, diameter = nodule_voxel_location
    rand_x, rand_y, rand_z = anchor_point

    crop_loc_x, crop_loc_y, crop_loc_z = (vox_x - rand_x, vox_y - rand_y, vox_z - rand_z)
    
    return (crop_loc_x, crop_loc_y, crop_loc_z, diameter // spacing[0])

def generateCrops(dataPath: str, cropsPerScan: int): 
    shortCount = 0
    totalCrops = 0

    for npyFile in glob.glob(os.path.join(dataPath, 'processed_scan', '*.npy')):   
        scan = CleanScan(npyPath=npyFile) 

        if scan.img.shape[0] < 97: 
            shortCount += 1
            continue

        crops = cropCube(scan=scan.img, numCubes=cropsPerScan)

        label = 0
        for c, anchor in crops: 
            if len(scan.annotations) == 0: 
                continue 

            for i in scan.annotations: 
                nodule_voxel_location  = worldToVoxel(world_point=i, world_origin=scan.origin, 
                                                      spacing=scan.spacing)
                vox_x, vox_y, vox_z, _ = nodule_voxel_location

                x0, y0, z0 = anchor

                if (vox_x in range(x0, x0 + 96)) and (vox_y in range(y0, y0 + 96)) and (vox_z in range(z0, z0 + 96)): 
                    crop_location = scanToCropNoduleLocation(anchor_point=anchor, 
                                                             nodule_voxel_location=nodule_voxel_location,
                                                             spacing=scan.spacing)
                                                            
                    label = 1
                    break
            
            outpath = os.path.join(dataPath, 'dataset', f'{scan.scanId}_{str(totalCrops)}.npz')
            np.savez_compressed(file=outpath,
                                img=[c,],
                                label=label)
            
            print(f'wrote crop to {outpath}.')
            totalCrops += 1
            
    print(f'shortCount: {shortCount}')
    print(f'totalCrops: {totalCrops}')
