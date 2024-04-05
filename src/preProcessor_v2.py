import os
import glob

from util.util import scanPathToId, generateCrops
from dataModels.scan import Scan

# TODO: use argparse! 

scans = True
crops = True 

dataPath = 'data/' 
annotationPath = os.path.join(dataPath, 'annotations.csv')

def main():
    scanList = []
    cropList = []

    if scans:
        for i in range(1):              
            for mhdPath in glob.glob(os.path.join(dataPath, f'images/subset{i}', '*.mhd')):   
                scanId = scanPathToId(mhdPath)

                maskPath = os.path.join(dataPath, 'seg_lungs', f'{scanId}.mhd')
                outPath  = os.path.join(dataPath, 'processed_scan', f'{scanId}.npy')
                scanList.append(Scan(mhdPath=mhdPath, maskPath=maskPath, annotationPath=annotationPath, outputPath=outPath))

        print('finished processing scans.')
    
    if crops: 
        cropList = generateCrops(scanList=scanList, cropsPerScan=2, outputPath='crops')

if __name__ == "__main__": 
    main()
