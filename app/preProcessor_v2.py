import os
import glob

from src.util.util import scanPathToId
from src.dataModels.scan import RawScan 

# TODO: use argparse! 

dataPath = 'data/' 
annotationPath = os.path.join(dataPath, 'annotations.csv')

def main():
    rawScanList = []
    count = 0

    for i in range(1):              
        for mhdPath in glob.glob(os.path.join(dataPath, f'images/subset{i}', '*.mhd')):   
            if count == 45: 
                break

            scanId = scanPathToId(mhdPath)

            maskPath = os.path.join(dataPath, 'seg_lungs', f'{scanId}.mhd')
            outPath  = os.path.join(dataPath, 'processed_scan', f'{scanId}.npy')
            rawScanList.append(RawScan(mhdPath=mhdPath, maskPath=maskPath, annotationPath=annotationPath, outputPath=outPath))

            count += 1

    print('finished processing scans.')

if __name__ == "__main__": 
    main()
