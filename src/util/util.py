import matplotlib.pyplot as plt 
import numpy as np
import scipy
from scipy import ndimage
from skimage import morphology

i = 0

def show_slice(img: [], save=False): 
    plt.imshow(img, cmap='grey')
    
    plt.xticks([])
    plt.yticks([])

    if save: 
        global i
        plt.savefig('example_images/{}.png'.format(i))
        i += 1

    plt.show()

def show_comparison(origImg: [], newImg: []): 
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(12, 6))
    axes[0].imshow(origImg, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(newImg, cmap='gray')
    axes[1].axis('off') 
    
    global i
    plt.savefig('example_images/{}.png'.format(i))
    i += 1
    
    plt.show()

def show_contour(img, contours):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=1)

    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()

def window_image(img: list, window: int, level: int) -> []: 
    min_hu = level - (window // 2)
    max_hu = level + (window // 2)

    windowed_img = np.copy(img)
    windowed_img = np.clip(windowed_img, -1200, 600)

    return windowed_img
        
def is_contour_closed(ct):
    ct_start = np.array((ct[0, 1], ct[0, 0]))
    ct_end   = np.array((ct[-1, 1], ct[-1, 0]))

    distance = np.linalg.norm(ct_start - ct_end)

    if distance == 0: 
        return True
    else: 
        return False
    
def choose_contours(contours):
    lung_contours = []
    volumes       = []

    for ct in contours: 
        hull = scipy.spatial.ConvexHull(ct)

        if hull.volume > 2000 and is_contour_closed(ct): 
            lung_contours.append(ct)
            volumes.append(hull.volume)

    if len(lung_contours) == 2: 
        return lung_contours
    else: 
        vol, ct = (list(t) for t in zip(*sorted(zip(volumes, lung_contours))))
        lung_contours = lung_contours[0:2]

        return lung_contours
    
def rm_noise(img: list): 
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