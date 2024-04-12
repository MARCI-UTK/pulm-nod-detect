import os 
import numpy as np

class Crop(): 
    def __init__(self, scanId, img, label):
        self.scanId     = scanId
        self.img        = img
        self.label      = label

