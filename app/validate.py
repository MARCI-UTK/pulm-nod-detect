import os

dataPath = '/data/marci/dlewis37/luna16/' 

rawScanLen = 0
for i in range(10):              
    subsetPath = os.path.join(dataPath, f'scan/subset{i}')

    rawFiles = [f for f in os.listdir(subsetPath) if f.endswith('.mhd')]
    rawScanLen += len(rawFiles)

npyFiles = [f for f in os.listdir(os.path.join(dataPath, 'processed_scan')) if f.endswith('.npy')]
jsonFiles = [f for f in os.listdir(os.path.join(dataPath, 'processed_scan')) if f.endswith('.json')]

print(f'raw files: {rawScanLen}. npy files: {len(npyFiles)}. json files: {len(jsonFiles)}')

scanIdListNpy = [f.split('/')[-1][:-4] for f in npyFiles]
scanIdListJson = [f.split('/')[-1][:-5] for f in jsonFiles]


diff = [x for x in scanIdListNpy if x not in scanIdListJson]
print(diff)
print(len(diff))