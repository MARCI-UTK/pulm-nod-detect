import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .util import worldToVoxel
from ..dataModels.crop import Crop
from ..dataModels.scan import CleanScan


def cropCube(scan: np.array, numCubes: int) -> list: 
    crops = []

    for _ in numCubes:
        randX = np.random.randint(0, scan.shape[2] - 96)
        randY = np.random.randint(0, scan.shape[1] - 96)
        randZ = np.random.randint(0, scan.shape[0] - 96)

        crop = scan[randZ:randZ + 96, randY:randY + 96, randX:randX + 96]

        crops.append(crop)

    return crops

def scanToCropNoduleLocation(anchor_point, nodule_voxel_location): 
    vox_x, vox_y, vox_z, diameter = nodule_voxel_location
    rand_x, rand_y, rand_z = anchor_point

    crop_loc_x, crop_loc_y, crop_loc_z = (vox_x - rand_x, vox_y - rand_y, vox_z - rand_z)
    print(crop_loc_x, crop_loc_y, crop_loc_z)


def generateCrops(scanList: list, cropsPerScan: int, outputPath: str) -> list: 
    rv = []

    for s in scanList: 
        crops = cropCube(s, cropsPerScan)

        label = 0  
        for c in crops: 
            if len(s.annotations) == 0: 
                continue
            
            for i in s.annotations: 
                nodule_voxel_location  = worldToVoxel(world_point=i, world_origin=s.origin, spacing=s.spacing)
                vox_x, vox_y, vox_z, _ = nodule_voxel_location

                anchor_point = (c[2,0], c[1,0], c[0,0])
                x0, y0, z0   = anchor_point

                if (vox_x in range(x0, x0 + 96)) and (vox_y in range(y0, y0 + 96)) and (vox_z in range(z0, z0 + 96)): 
                    crop_location = scanToCropNoduleLocation(anchor_point=anchor_point, 
                                                             nodule_voxel_location=nodule_voxel_location)
                    
                    label = 1

                    ax = plt.gca()
                    ax.cla()

                    slice_idx = crop_location[2]
                    ax.imshow(c[slice_idx], cmap='gray')

                    circle = Circle((crop_location[0], crop_location[1]), crop_location[3], color='r', fill=False)
                    ax.add_patch(circle)

                    plt.title(s.scan_id)
                    plt.show
            
            rv.append(Crop(scanId=s.scanId, img=c, label=label, outputPath=outputPath))

    return rv

def scanListFromNpy(path: str) -> list: 
    rv = []

    for f in os.listdir(path):
        rv.append(CleanScan(path))