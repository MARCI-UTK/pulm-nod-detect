import os
import glob
import numpy as np 
from tqdm import tqdm
import SimpleITK as sitk
from src.dataModels.scan import windowImage

from src.util.util import scanPathToId
from src.dataModels.scan import RawScan 

# Paths to the LUNA16 dataset directory  
dataPath = '/data/marci/luna16/' 

# Dataset annotations (.csv)
annotationPath = os.path.join(dataPath, 'annotations.csv')

def resample_image(image: sitk.Image, target_spacing: tuple): 
    # Resize spacing to 1mm^3

    # (x, y, z) spacing in mm
    original_spacing = image.GetSpacing()  

    # (width, height, depth)
    original_size = image.GetSize() 

    # Compute new size to maintain aspect ratio
    new_size = [
        int(round(osz * osp / tsp)) 
        for osz, osp, tsp in zip(original_size, original_spacing, target_spacing)
    ]

    interpolator = sitk.sitkBSpline

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetInterpolator(interpolator)

    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    
    image = resampler.Execute(image)

    return image


"""
Iteratre through all 10 subsets of CT scans and perform preprocessing operations on each scan
Save the processed scan as a .npy file using the scan's ID from the data directory. 
"""
def main():

    for j in range(10):     
        with tqdm(glob.glob(os.path.join(dataPath, f'img/subset{j}', '*.mhd'))) as pbar:          
            for mhdPath in pbar:   
                scanId = scanPathToId(mhdPath)

                # Segmentation mask 
                maskPath = os.path.join(dataPath, 'segmentation_masks', f'{scanId}.mhd')

                # Image output path 
                npyPath  = os.path.join(dataPath, 'processed_scan', f'{scanId}.npy')

                # Image metadata output 
                jsonPath = os.path.join(dataPath, 'processed_scan', f'{scanId}.json')

                # Read scan 
                img = sitk.ReadImage(fileName=mhdPath)

                # Resize spacing to 1mm^3
                target_spacing = (1.0, 1.0, 1.0) 
                scan = resample_image(image=scan, target_spacing=target_spacing)
                scan = sitk.GetArrayFromImage(image=img)

                # Read mask and resize spacing 
                mask = sitk.ReadImage(maskPath)
                mask = resample_image(image=mask, target_spacing=target_spacing)
                mask = sitk.GetArrayFromImage(image=mask)
                mask[mask > 0] = 255

                # Pre-processing 
                cleaned_scan = []

                # Iterate through each slice in scan 
                for i in range(len(scan)):
                    scanSlice = scan[i]
                    maskSlice = mask[i]

                    # Window and level Hounsfield range 
                    windowedScan = windowImage(img=scanSlice, window=1500, level=-600) 

                    # Apply the mask to scan 
                    maskHighVals = (maskSlice == 0)

                    final = np.copy(windowedScan)
                    final[maskHighVals] = np.min(windowedScan)

                    cleaned_scan.append(final) 

                pbar.set_postfix(subset=j)

if __name__ == "__main__": 
    main()