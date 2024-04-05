import os 
import numpy as np

class Crop(): 
    def __init__(self, scanId, img, label, outputPath):
        self.scanId     = scanId
        self.img        = img
        self.label      = label
        self.outputPath = outputPath

    def writeCrop(self): 
        data = np.array([self.img, self.label])

        outFile = os.path.join(self.outputPath, f'{self.scanId}.npy')
        np.save(outFile, data)
        print(f'wrote to {outFile}.')
