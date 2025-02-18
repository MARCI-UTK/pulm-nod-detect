import os
import glob
import concurrent
import concurrent.futures
from tqdm import tqdm

from src.util.util import scanPathToId
from src.dataModels.scan import RawScan 

# TODO: use argparse! 

# Paths to the LUNA16 dataset directory  
dataPath = '/data/marci/luna16/' 

# Dataset annotations (.csv)
annotationPath = os.path.join(dataPath, 'csv', 'annotations.csv')

def preproccess_scan(scan_path):
    scanId = scanPathToId(scan_path)

    # Segmentation mask 
    maskPath = os.path.join(dataPath, 'segmentation_masks', f'{scanId}.mhd')

    # Image output path 
    npyPath  = os.path.join(dataPath, 'processed_scan', f'{scanId}.npz')

    # Image metadata output 
    jsonPath = os.path.join(dataPath, 'processed_scan', f'{scanId}.json')

    # Use RawScan class to do preprocessing 
    RawScan(mhdPath=scan_path, maskPath=maskPath, 
            annotationPath=annotationPath, 
            npyPath=npyPath, jsonPath=jsonPath)

"""
Iteratre through all 10 subsets of CT scans and perform preprocessing operations on each scan
Save the processed scan as a .npy file using the scan's ID from the data directory. 
"""
def main():

    for i in range(10): 
        
        img_paths = glob.glob(os.path.join(dataPath, f'img/subset{i}', '*.mhd')) 
        with tqdm(img_paths) as pbar: 
            for path in pbar:
                try: 
                    preproccess_scan(path)
                except: 
                    continue
                
                pbar.update()
        """
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

            pbar.update()
        """

    print('finished preprocessing raw scans.')

if __name__ == "__main__": 
    main()
