import SimpleITK as sitk
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
import pandas as pd 
import numpy as np
import os

def get_example_files(n: int) -> list: 
    sub0_path = 'dataset/subset0/'
    sub0_files = []
    for f in os.listdir(sub0_path): 
        if f.endswith('.mhd'): 
            sub0_files.append(sub0_path + f)

    example_imgs = []
    example_idxs = np.random.randint(0, len(sub0_files) - 1, size=10)

    for i in example_idxs: 
        example_imgs.append(sub0_files[i])

    return example_imgs

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

def get_scan_nodule_locations(scan_id: str, annotations: pd.DataFrame) -> list: 
    scan_annotations = annotations[annotations['seriesuid'] == scan_id]

    nodule_locations = []
    for _, row in scan_annotations.iterrows(): 
        loc = (row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm'])
        nodule_locations.append(loc)
    
    return(nodule_locations)

def plot_nodules(scan_id, scan, locations): 

    for idx, i in enumerate(locations): 
        ax = plt.gca()
        ax.cla()

        slice_idx = i[2]
        ax.imshow(scan[slice_idx], cmap='gray')

        circle = Circle((i[0], i[1]), i[3], color='r', fill=False)
        ax.add_patch(circle)

        plt.title(scan_id)
        plt.savefig('example_images/imgs_with_labels/{}_nodule{}.png'.format(scan_id, idx))

files = get_example_files(10)
annotations = pd.read_csv('dataset/annotations.csv')

for f in files: 
    img = sitk.ReadImage(f)
    
    origin_point = img.GetOrigin()
    spacing = img.GetSpacing()
    scan_id = f.split('/')[2][0:-4]

    nodule_locations = get_scan_nodule_locations(scan_id=scan_id, annotations=annotations)

    if len(nodule_locations) == 0: 
        continue

    pre_processed_scan = np.load('dataset/processed_scan/{}.npy'.format(scan_id))

    nodule_voxel_coords = []
    for i in nodule_locations: 
        vox_coord = world_to_voxel(i, origin_point, spacing)
        nodule_voxel_coords.append(vox_coord)
        
    img_arr = sitk.GetArrayFromImage(img)
    plot_nodules(scan_id=scan_id, scan=pre_processed_scan, locations=nodule_voxel_coords)