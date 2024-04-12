import os
from src.util.crop_util import generateCrops

dataPath = '/data/marci/dlewis37/luna16/'

def main():  
    cropList = generateCrops(dataPath=dataPath, cropsPerScan=2)
    print(len(cropList))

    return 

if __name__ == "__main__": 
    main() 
