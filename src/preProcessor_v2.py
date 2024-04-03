import SimpleITK as sitk 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import skimage
import scipy
import os
from pathlib import Path
from util.util import *

### IN NUMPY AN ORDERED PAIR WILL BE (Y, X) ###

datasetPath = 'dataset/'

def preProcessScan(scan: list, mask: list) -> list: 
    processed_scan = []

    for i in range(len(scan)):
        scanSlice = scan[i]
        maskSlice = mask[i]

        windowedScan = window_image(img=scanSlice, window=600, level=-1200) 
        normalizedPixelSlice = (windowedScan // 256).astype('uint8')

        maskedScan = normalizedPixelSlice * maskSlice
        maskHighVals = (maskedScan == 0)

        final = np.copy(scanSlice)
        final[maskHighVals] = 0

        processed_scan.append(final) 

    return processed_scan

def main():  
    for i in range(1): 
        subPath = datasetPath + 'subset' + str(i) + '/'
        for f in os.listdir(subPath): 
            
            if f.endswith('.mhd'): 
                scanPath = subPath + f
                maskPath = datasetPath + 'seg_lungs/' + f

                rawImg = sitk.ReadImage(scanPath)
                imgArr = sitk.GetArrayFromImage(rawImg)

                rawMask = sitk.ReadImage(maskPath)
                maskArr = sitk.GetArrayFromImage(rawMask)
                
                newScan = preProcessScan(scan=imgArr, mask=maskArr)
                
                outPath  = datasetPath + 'processed_scan/' + Path(f).stem + '.npy'
                np.save(outPath, newScan)
                print('wrote to ' + outPath)

                # REMOVE THIS 
                break
            
if __name__ == "__main__": 
    main()
