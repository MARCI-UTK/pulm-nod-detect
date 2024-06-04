import matplotlib
matplotlib.use('Agg')

import os 
import glob
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .util import worldToVoxel
from ..dataModels.crop import Crop
from ..dataModels.scan import CleanScan

def noduleLocationToBb(location, img): 
    theta = math.pi / 4.0  
    x, y, z, d = location

    w = math.cos(theta) * d
    h = math.sin(theta) * d

    x1 = int(x - (w // 2)) - 1
    y1 = int(y - (h // 2)) - 1
    x2 = int(x + (w // 2)) + 1
    y2 = int(y + (h // 2)) + 1

    #imgBb = cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
    imgBb = img

    pltCoords = (x1, y1, w + 2, h + 2)

    return (imgBb, pltCoords)

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

        for c, anchor in crops: 
            label = 0

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

                    bb, rectCoords = noduleLocationToBb(crop_location, c[crop_location[2]])

                    xy = (rectCoords[0], rectCoords[1])
                    w, h  = rectCoords[2], rectCoords[3]

                    ax = plt.gca()
                    ax.cla()

                    ax.imshow(c[crop_location[2]], cmap='gray')

                    rect = Rectangle(xy=xy, width=w, height=h, color='r', fill=False)
                    ax.add_patch(rect)

                    #plt.imshow(c[crop_location[2]], cmap='gray')
                    plt.savefig('test_imgs/bb.png')

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
