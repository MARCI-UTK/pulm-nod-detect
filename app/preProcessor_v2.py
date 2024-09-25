import os
import glob

from src.util.util import scanPathToId
from src.dataModels.scan import RawScan 

# TODO: use argparse! 

# Paths to the LUNA16 dataset directory  
dataPath = '/data/marci/luna16/' 

# Dataset annotations (.csv)
annotationPath = os.path.join(dataPath, 'annotations.csv')

"""
Iteratre through all 10 subsets of CT scans and perform preprocessing operations on each scan
Save the processed scan as a .npy file using the scan's ID from the data directory. 
"""
def main():

    for i in range(10):              
        for mhdPath in glob.glob(os.path.join(dataPath, f'img/subset{i}', '*.mhd')):   
            scanId = scanPathToId(mhdPath)

            # Segmentation mask 
            maskPath = os.path.join(dataPath, 'segmentation_masks', f'{scanId}.mhd')

            # Image output path 
            npyPath  = os.path.join(dataPath, 'processed_scan', f'{scanId}.npy')

            # Image metadata output 
            jsonPath = os.path.join(dataPath, 'processed_scan', f'{scanId}.json')

            # Use RawScan class to do preprocessing 
            RawScan(mhdPath=mhdPath, maskPath=maskPath, 
                    annotationPath=annotationPath, 
                    npyPath=npyPath, jsonPath=jsonPath)

    print('finished preprocessing raw scans.')

if __name__ == "__main__": 
    main()
