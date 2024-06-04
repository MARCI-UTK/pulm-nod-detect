import os 
import glob
import PIL 
from PIL import Image
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from src.util.util import windowImage

scanDir = '/data/marci/dlewis37/luna16/scan/'
maskDir = '/data/marci/dlewis37/luna16/mask/'

for i in range(2,10):
    subDir = os.path.join(scanDir, f'subset{i}')

    for s in glob.glob(os.path.join(subDir, '*.mhd')):

        mhd = sitk.ReadImage(s)
        arr = sitk.GetArrayFromImage(mhd)

        suid = s.split('/')[-1][0:-4]
        mask = sitk.ReadImage(os.path.join(maskDir, f'{suid}.mhd'))
        mask = sitk.GetArrayFromImage(mask)

        #arr *= mask

        sl = arr[len(arr) // 2]
        windowed = windowImage(sl, 1500, -600)

        mskSl = mask[len(mask) // 2]

        windowed *= mskSl

        #im = Image.fromarray(sl)
        #msk = Image.fromarray(mskSl)
        
        sl = np.array(sl)
        
        high = np.max(sl)
        low  = np.min(sl)

        tmp = np.copy(sl)

        """
        for ridx, r in enumerate(sl): 
            for cidx, px in enumerate(r): 
                tmp[ridx][cidx] = ((px - low) * 255) / (high - low)

        """
        plt.imsave('test.png', windowed, cmap='gray')
        plt.imsave('scan.png', sl, cmap='gray')


        #tmp = Image.fromarray(sl)
        #arr = np.array(tmp)
        #print(np.min(arr), np.max(arr))
        #plt.imsave('mask.png', mskSl, cmap='gray')

        break 
    break