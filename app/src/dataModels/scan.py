import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

from ..util.util import scanPathToId, windowImage

class RawScan(): 
    def __init__(self, mhdPath, maskPath, annotationPath, outputPath): 
        self.mhdPath = mhdPath
        self.maskPath = maskPath
        self.outputPath = outputPath
        self.annotationPath = annotationPath
        
        self.scanId      = None
        self.origin      = None
        self.spacing     = None 
        self.annotations = None 
        self.rawImg      = None 
        self.cleanImg    = None 
        self.mask        = None

        self.processScan()
        self.writeProcessedScan()

    def processScan(self):
        self.readMhd()
        self.readMask()
        self.cleanScan()

    def readMhd(self):
        path = self.mhdPath

        img = sitk.ReadImage(fileName=path)

        self.scanId = scanPathToId(path=path)
        self.rawImg = sitk.GetArrayFromImage(image=img)
        self.origin = img.GetOrigin()
        self.spacing = img.GetSpacing()

    def readMask(self): 
        mask = sitk.ReadImage(fileName=self.maskPath)
        self.mask = sitk.GetArrayFromImage(image=mask)

    def cleanScan(self): 
        cleanScan = []
        scan = self.rawImg

        for i in range(len(scan)):
            scanSlice = scan[i]
            maskSlice = self.mask[i]

            windowedScan = windowImage(img=scanSlice, window=600, level=-1200) 
            normalizedPixelSlice = (windowedScan // 256).astype('uint8')

            maskedScan = normalizedPixelSlice * maskSlice
            maskHighVals = (maskedScan == 0)

            final = np.copy(scanSlice)
            final[maskHighVals] = 0

            cleanScan.append(final) 

        self.cleanImg = cleanScan

    def get_scan_nodule_locations(self): 
        annotations = pd.read_csv(self.annotationPath)
        scan_annotations = annotations[annotations['seriesuid'] == self.scanId]

        nodule_locations = []
        for _, row in scan_annotations.iterrows(): 
            loc = (row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm'])
            nodule_locations.append(loc)
        
        self.annotations = nodule_locations
    
    def writeProcessedScan(self): 
        np.save(self.outputPath, self.cleanImg)
        print(f'wrote to {self.outputPath}.')

class CleanScan(): 
    def __init__(self, npyPath): 
        self.scanId = scanPathToId(npyPath)
        self.img    = np.load(npyPath)

        self.origin = None
        self.spacing = None
        self.annotations = None 

        self.readMhdData()
        self.get_scan_nodule_locations()

    def readMhdData(self): 
        reader = sitk.ImageFileReader()

        #mhdPath = os.path.join('data', 'images', 'subset0', f'{self.scanId}.mhd')
        mhdPath = f'/data/marci/dlewis37/luna16/scan/subset0/{self.scanId}.mhd'
        reader.SetFileName(fn=mhdPath)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        self.origin  = reader.GetOrigin()
        self.spacing = reader.GetSpacing()

    def get_scan_nodule_locations(self): 
        annotations = pd.read_csv('/data/marci/dlewis37/luna16/csv/annotations.csv')
        scan_annotations = annotations[annotations['seriesuid'] == self.scanId]

        nodule_locations = []
        for _, row in scan_annotations.iterrows(): 
            loc = (row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm'])
            nodule_locations.append(loc)
        
        self.annotations = nodule_locations
