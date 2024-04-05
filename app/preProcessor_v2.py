import os
import glob

from src.util.util import scanPathToId
from src.dataModels.scan import RawScan 

# TODO: use argparse! 

dataPath = '/data/marci/dlewis37/luna16/' 
annotationPath = os.path.join(dataPath, 'annotations.csv')

def main():
    rawScanList = []
   
    for i in range(1):              
        for mhdPath in glob.glob(os.path.join(dataPath, f'scan/subset{i}', '*.mhd')):   
            scanId = scanPathToId(mhdPath)

            maskPath = os.path.join(dataPath, 'mask', f'{scanId}.mhd')
            outPath  = os.path.join(dataPath, 'processed_scan', f'{scanId}.npy')
            rawScanList.append(RawScan(mhdPath=mhdPath, maskPath=maskPath, annotationPath=annotationPath, outputPath=outPath))

    print('finished processing scans.')

if __name__ == "__main__": 
    main()
