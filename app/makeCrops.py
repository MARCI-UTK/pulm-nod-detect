import os
from src.util.crop_util import scanListFromNpy, generateCrops

dataPath = 'data/'

def main():  
    cleanScanList = scanListFromNpy(os.path.join(dataPath, 'processed_scan'))
    cropList = generateCrops(scanList=cleanScanList, cropsPerScan=2, outputPath=os.path.join(dataPath, 'crop'))

    return 

if __name__ == "__main__": 
    main() 