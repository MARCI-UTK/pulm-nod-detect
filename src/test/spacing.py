import SimpleITK as sitk 
import matplotlib.pyplot as plt

def world_to_voxel(world_point: tuple, world_origin: tuple, spacing: tuple) -> tuple: 
    world_x, world_y, world_z       = world_point
    origin_x, origin_y, origin_z    = world_origin
    spacing_x, spacing_y, spacing_z = spacing

    voxel_x = (world_x - origin_x) // spacing_x
    voxel_y = (world_y - origin_y) // spacing_y
    voxel_z = (world_z - origin_z) // spacing_z

    voxel_point = (voxel_x, voxel_y, voxel_z)

    return(voxel_point)

img = sitk.ReadImage('dataset/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.108197895896446896160048741492.mhd')

# The (x, y, z) point in physical space where the voxel (0, 0, 0) is located 
origin_point = img.GetOrigin()

# The physical distance between each voxel coordinate 
# This is not consistent for the x and y directions but usually is for z 
spacing = img.GetSpacing()

test_point = (-100.5679445, 67.26051683, -231.816619)
test_point_voxel_coord = world_to_voxel(world_point=test_point, world_origin=origin_point, spacing=spacing)

img_arr = sitk.GetArrayFromImage(img)

test_slice = img_arr[int(test_point_voxel_coord[2])]

plt.imshow(test_slice, cmap='gray')
#plt.plot(int(test_point_voxel_coord[0]), int(test_point_voxel_coord[1]), 'bo')
plt.show()