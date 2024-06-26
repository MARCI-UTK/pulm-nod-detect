import os
from src.util.crop_util import generateCrops

dataPath = '/data/marci/dlewis37/luna16/'

def main():  
    generateCrops(dataPath=dataPath)

    return 

if __name__ == "__main__": 
    main() 
