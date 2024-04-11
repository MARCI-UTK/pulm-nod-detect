import os

dataPath = '/data/marci/dlewis37/luna16/' 

rawScanLen = 0
for i in range(10):              
    subsetPath = os.path.join(dataPath, f'scan/subset{i}')

    rawFiles = [f for f in os.listdir(subsetPath) if f.endswith('.mhd')]
    print(f'{subsetPath} length: {len(os.listdir(subsetPath))}')
    rawScanLen += len(rawFiles)

processedScanLen = len(os.listdir(os.path.join(dataPath, 'processed_scan')))
print(f'raw files: {rawScanLen}. processed files: {processedScanLen}.')