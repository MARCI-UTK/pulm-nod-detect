import os
import numpy as np 
import pandas as pd
import SimpleITK as sitk 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

annotations = pd.read_csv('dataset/annotations.csv')

def get_origin_and_spacing(scan_id): 
    img = sitk.ReadImage('dataset/subset0/{}.mhd'.format(scan_id))

    origin = img.GetOrigin()
    spacing = img.GetSpacing()

    return (origin, spacing)

def get_example_files(n: int) -> list: 
    path = 'dataset/processed_scan/'
    files = []
    for f in os.listdir(path): 
        files.append(path + f)

    example_imgs = []
    example_idxs = np.random.randint(0, len(files) - 1, size=n)

    for i in example_idxs: 
        example_imgs.append(files[i])

    return example_imgs

def get_scan_nodule_locations(scan_id: str, annotations: pd.DataFrame) -> list: 
    scan_annotations = annotations[annotations['seriesuid'] == scan_id]

    nodule_locations = []
    for _, row in scan_annotations.iterrows(): 
        loc = (row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm'])
        nodule_locations.append(loc)
    
    return(nodule_locations)

def world_to_voxel(world_point: tuple, world_origin: tuple, spacing: tuple) -> tuple: 
    world_x, world_y, world_z, diameter = world_point
    origin_x, origin_y, origin_z        = world_origin
    spacing_x, spacing_y, spacing_z     = spacing

    voxel_x = (world_x - origin_x) // spacing_x
    voxel_y = (world_y - origin_y) // spacing_y
    voxel_z = (world_z - origin_z) // spacing_z

    voxel_diameter = diameter // spacing_x

    voxel_point = (int(voxel_x), int(voxel_y), int(voxel_z), int(voxel_diameter))

    return(voxel_point)

def scan_nodule_location_to_crop_nodule_location(anchor_point, nodule_voxel_location): 
    vox_x, vox_y, vox_z, diameter = nodule_voxel_location
    rand_x, rand_y, rand_z = anchor_point

    crop_loc_x, crop_loc_y, crop_loc_z = (vox_x - rand_x, vox_y - rand_y, vox_z - rand_z)
    print(crop_loc_x, crop_loc_y, crop_loc_z)

    return (crop_loc_x, crop_loc_y, crop_loc_z, diameter)

def check_crop_for_nodules(crop, anchor_point, origin_point, spacing, nodule_locations): 
    if len(nodule_locations) == 0: 
        return 0 
    
    for idx, i in enumerate(nodule_locations): 
        nodule_voxel_location = world_to_voxel(world_point=i, world_origin=origin_point, spacing=spacing)
        vox_x, vox_y, vox_z, _ = nodule_voxel_location
        rand_x, rand_y, rand_z = anchor_point

        if (vox_x in range(rand_x, rand_x + 96)) and (vox_y in range(rand_y, rand_y + 96)) and (vox_z in range(rand_z, rand_z + 96)): 
            crop_location = scan_nodule_location_to_crop_nodule_location(anchor_point=anchor_point, 
                                                                         nodule_voxel_location=nodule_voxel_location)
            
            ax = plt.gca()
            ax.cla()

            slice_idx = crop_location[2]
            ax.imshow(crop[slice_idx], cmap='gray')

            circle = Circle((crop_location[0], crop_location[1]), crop_location[3], color='r', fill=False)
            ax.add_patch(circle)

            plt.title(scan_id)
            plt.savefig('example_images/crops_with_labels/{}_{}.png'.format(scan_id, idx))
        
files = get_example_files(50)
for idx, f in enumerate(files): 
    data = np.load(f)

    rand_x = np.random.randint(0, data.shape[2] - 96)
    rand_y = np.random.randint(0, data.shape[1] - 96)
    rand_z = np.random.randint(0, data.shape[0] - 96)

    random_crop = data[rand_z:rand_z + 96, rand_y:rand_y + 96, rand_x:rand_x + 96]

    scan_id = f.split('/')[2][0:-4]
    nodule_locations = get_scan_nodule_locations(scan_id=scan_id, annotations=annotations)

    origin, spacing = get_origin_and_spacing(scan_id=scan_id)

    check_crop_for_nodules(crop=random_crop, anchor_point=(rand_x, rand_y, rand_z), 
                           origin_point=origin, spacing=spacing, nodule_locations=nodule_locations)

    """
    plt.imshow(random_crop[50], cmap='gray')
    plt.savefig('example_images/random_crops/{}_{}.png'.format(scan_id, idx))
    plt.show()
    """