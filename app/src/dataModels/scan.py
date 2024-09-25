import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import json

import matplotlib.pyplot as plt 

from ..util.util import scanPathToId, windowImage

# Class to handle scan preprocessing operations 
class RawScan(): 
    def __init__(self, mhdPath, maskPath, annotationPath, 
                 npyPath, jsonPath):

        # Save input data 
        self.mhdPath = mhdPath
        self.maskPath = maskPath
        self.npyPath = npyPath
        self.jsonPath = jsonPath
        self.annotationPath = annotationPath

        # Data read from .mhd 
        self.scanId      = None
        self.origin      = None
        self.spacing     = None 
        
        # Nodule locations within scan 
        self.annotations = None 

        # Image arrays 
        self.rawImg      = None 
        self.cleanImg    = None 
        self.mask        = None

        self.processScan()
        self.writeProcessedScan()

    # Read scan and mask, pass to processing function 
    def processScan(self):
        self.readMhd()
        self.readMask()
        self.cleanScan()

    # Read scan and extract metadata
    def readMhd(self):
        path = self.mhdPath

        img = sitk.ReadImage(fileName=path)

        self.scanId = scanPathToId(path=path)
        self.rawImg = sitk.GetArrayFromImage(image=img)
        self.origin = img.GetOrigin()
        self.spacing = img.GetSpacing()

    # Read segmenation mask 
    def readMask(self): 
        mask = sitk.ReadImage(fileName=self.maskPath)
        mask = sitk.GetArrayFromImage(image=mask)
        
        mask[mask > 0] = 255

        self.mask = mask

    # Perform preprocessing operations on scan 
    def cleanScan(self): 
        cleanScan = []
        scan = self.rawImg

        # Iterate through each slice in scan 
        for i in range(len(scan)):
            scanSlice = scan[i]
            maskSlice = self.mask[i]

            # Window and level Hounsfield range 
            windowedScan = windowImage(img=scanSlice, window=1500, level=-600) 

            maskHighVals = (maskSlice == 0)

            final = np.copy(windowedScan)
            final[maskHighVals] = np.min(windowedScan)

            cleanScan.append(final) 

        self.cleanImg = cleanScan

    # Get nodule locations corresponding to current scan ID 
    def get_scan_nodule_locations(self): 
        annotations = pd.read_csv(self.annotationPath)
        scan_annotations = annotations[annotations['seriesuid'] == self.scanId]

        nodule_locations = []
        for _, row in scan_annotations.iterrows(): 
            loc = (row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm'])
            nodule_locations.append(loc)
        
        self.annotations = nodule_locations
    
    # Write processed image as .npy and metadata as .json files 
    def writeProcessedScan(self): 
        metadata = {
            'origin': self.origin, 
            'spacing': self.spacing
        }

        f = open(self.jsonPath, 'w')
        json.dump(metadata, f)
        f.close()

        np.save(self.npyPath, self.cleanImg)
        print(f'wrote data for {self.scanId}.')

# Class that holds preprocessed scan and its metadata for crop generation
class CleanScan(): 
    def __init__(self, npyPath): 
        self.scanId = scanPathToId(npyPath)
        self.img    = np.load(npyPath)

        self.origin = None
        self.spacing = None
        self.annotations = None 

        self.readMetadata(npyPath=npyPath)
        self.get_scan_nodule_locations()

    def readMetadata(self, npyPath): 
        metadataPath = npyPath[:-4] + '.json'
        
        with open(metadataPath, 'r') as f:
            metadata = json.load(f)

        self.origin = metadata['origin']
        self.spacing = metadata['spacing']

    def get_scan_nodule_locations(self): 
        annotations = pd.read_csv('/data/marci/luna16/csv/annotations.csv')
        scan_annotations = annotations[annotations['seriesuid'] == self.scanId]

        nodule_locations = []
        for _, row in scan_annotations.iterrows(): 
            loc = [row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm']]
            nodule_locations.append(loc)
        
        self.annotations = nodule_locations
