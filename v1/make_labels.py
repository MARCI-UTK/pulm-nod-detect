import numpy as np
import pandas as pd
import os 

annotations = pd.read_csv('dataset/annotations.csv')

def get_scan_nodule_locations(scan_id: str, annotations: pd.DataFrame) -> list: 
    scan_annotations = annotations[annotations['seriesuid'] == scan_id]

    nodule_locations = []
    for _, row in scan_annotations.iterrows(): 
        loc = (row['coordX'], row['coordY'], row['coordZ'])
        nodule_locations.append(loc)
    
    return(nodule_locations)

for f in os.listdir('dataset/processed_scan'): 
    scan_id = f[0:-4]

    nodule_locations = get_scan_nodule_locations
