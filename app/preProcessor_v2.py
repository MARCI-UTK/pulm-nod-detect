import os
import glob

from src.util.util import scanPathToId
from src.dataModels.scan import RawScan 

# TODO: use argparse! 

dataPath = '/data/marci/dlewis37/luna16/' 
annotationPath = os.path.join(dataPath, 'annotations.csv')

def main():

    for i in range(10):              
        for mhdPath in glob.glob(os.path.join(dataPath, f'scan/subset{i}', '*.mhd')):   
            scanId = scanPathToId(mhdPath)

            maskPath = os.path.join(dataPath, 'mask', f'{scanId}.mhd')
            npyPath  = os.path.join(dataPath, 'processed_scan', f'{scanId}.npy')
            jsonPath = os.path.join(dataPath, 'processed_scan', f'{scanId}.json')
            RawScan(mhdPath=mhdPath, maskPath=maskPath, 
                    annotationPath=annotationPath, 
                    npyPath=npyPath, jsonPath=jsonPath)

    print('finished preprocessing raw scans.')

if __name__ == "__main__": 
    main()
