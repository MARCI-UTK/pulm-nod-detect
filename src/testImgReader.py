import SimpleITK as sitk 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 

from scipy import ndimage
from skimage import morphology

def show_slice(img: []): 
    plt.imshow(img, cmap='grey')
    plt.show()

def window_image(img: [], window: int, level: int) -> []: 
    min_hu = level - (window // 2)
    max_hu = level + (window // 2)

    windowed_img = np.copy(img)
    windowed_img[windowed_img < min_hu] = min_hu
    windowed_img[windowed_img > max_hu] = max_hu

    return windowed_img

def rm_noise(img: []): 
    segmentation = morphology.dilation(img, np.ones(shape=(1, 1)))
    labels, _ = ndimage.label(segmentation)
    
    label_count = np.bincount(labels.ravel().astype(int))
    label_count[0] = 0

    mask = labels == label_count.argmax()

    mask = morphology.dilation(mask, np.ones((1, 1)))
    mask = ndimage.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))

    maskedImg = mask * img
    
    return maskedImg

def crop(img: []) -> []: 
    mask = img == 0

    coords = np.array(np.nonzero(~mask))
    topLeft = np.min(coords, axis=1)
    bottomRight = np.max(coords, axis=1)

    croppedImg = img[topLeft[0]:bottomRight[0], 
                     topLeft[1]:bottomRight[1]]
    
    return croppedImg

def pad(img: [], new_height=512, new_width=512) -> []: 
    height, width = img.shape

    final_img = np.zeros((new_height, new_width))

    pad_left = int((new_width - width) // 2)
    pad_top  = int((new_height - height) // 2)

    final_img[pad_top:pad_top + height, pad_left: pad_left + width] = img
    return img

def show_comparison(origImg: [], newImg: []): 
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(20, 10))
    axes[0].imshow(origImg, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(newImg, cmap='gray')
    axes[1].axis('off') 

    plt.show()

fname = 'dataset/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd'

rawImg = sitk.ReadImage(fname)
imgArr = sitk.GetArrayFromImage(rawImg)

"""
print('dimensions: {}'.format(img.GetDimension()))
print('size:       {}'.format(img.GetSize()))
print('origin:     {}'.format(img.GetOrigin()))
print('spacing:    {}'.format(img.GetSpacing()))
print('direction:  {}'.format(img.GetDirection()))
print('width:      {}'.format(img.GetWidth()))
print('height:     {}'.format(img.GetHeight()))
print('depth:      {}'.format(img.GetDepth()))
"""

# In the original img (after sitk.ReadImage()), the shape is in (x, y, z) order
# In the imgArr (after sitk.GetArrayFromImage()), the shape is in (z, y, x) order

level  = -600
window = 1500 

test = window_image(img=imgArr[80], window=window, level=level)
test = rm_noise(test)
test = crop(test)
test = pad(test)

show_slice(test)