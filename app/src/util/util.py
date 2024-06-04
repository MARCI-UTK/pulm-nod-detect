import numpy as np

def scanPathToId(path: str) -> str: 
    return path.split('/')[-1][0:-4]

def windowImage(img: list, window: int, level: int) -> list: 
    min_hu = level - (window // 2)
    max_hu = level + (window // 2)

    windowed_img = np.copy(img)
    windowed_img = np.clip(windowed_img, min_hu, max_hu)

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