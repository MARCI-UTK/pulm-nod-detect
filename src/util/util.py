import numpy as np
from dataModels.crop import Crop

import matplotlib.pyplot as plt 
from matplotlib.patches import Circle

def scanPathToId(path: str) -> str: 
    return path.split('/')[-1][0:-4]

def windowImage(img: list, window: int, level: int) -> list: 
    min_hu = level - (window // 2)
    max_hu = level + (window // 2)

    windowed_img = np.copy(img)
    windowed_img = np.clip(windowed_img, -1200, 600)

    return windowed_img

def worldToVoxel(world_point: tuple, world_origin: tuple, spacing: tuple) -> tuple: 
    world_x, world_y, world_z, diameter = world_point
    origin_x, origin_y, origin_z        = world_origin
    spacing_x, spacing_y, spacing_z     = spacing

    voxel_x = (world_x - origin_x) // spacing_x
    voxel_y = (world_y - origin_y) // spacing_y
    voxel_z = (world_z - origin_z) // spacing_z

    voxel_diameter = diameter // spacing_x

    voxel_point = (int(voxel_x), int(voxel_y), int(voxel_z), int(voxel_diameter))

    return(voxel_point)

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