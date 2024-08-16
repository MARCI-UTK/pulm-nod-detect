from torchio.transforms import RandomAffine, RandomFlip
from torch.utils.data import Dataset, DataLoader
import numpy as np  
import os
import torch 
import matplotlib.pyplot as plt

dataPath = '/data/marci/dlewis37/luna16/'

class AugmentTest(Dataset):
    def __init__(self, img_paths, label_paths):
        self.img_paths = img_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        x = np.load(self.img_paths[index])
        y = np.load(self.label_paths[index])
        
        fname = self.img_paths[index]
        
        x = torch.from_numpy(x['img']).float()
        labels = torch.from_numpy(y['labels']).float()
        locs = torch.from_numpy(y['locs']).float()

        flip = RandomFlip(axes=(2,))
        affinine = RandomAffine(scales=(0.75, 1.25), degrees=0)

        flipped_x = flip(x)
        affinine_x = affinine(x)
        f_and_a_x = affinine(flipped_x)

        return x, flipped_x, affinine_x, f_and_a_x, locs, labels
    
img_paths = [os.path.join(dataPath, 'dataset', f) for f in os.listdir(os.path.join(dataPath, 'dataset'))]
label_paths = [os.path.join(dataPath, 'rpn_labels', f) for f in os.listdir(os.path.join(dataPath, 'rpn_labels'))]

dataset = AugmentTest(img_paths=img_paths, label_paths=label_paths)
loader = DataLoader(
    dataset=dataset,
    batch_size=1, 
    shuffle=True
)

img_cnt = 0
for idx, data in enumerate(loader):
    x, flipped_x, affinine_x, f_and_a_x, gt_box, label = data

    if gt_box.sum() == 0: 
        continue
    
    """
    f, ax = plt.subplots(2, 2)
    ax[0, 0] = plt.imshow(x[0, 0, 50], cmap=plt.bone())
    ax[0, 1] = plt.imshow(flipped_x[0, 0, 50], cmap=plt.bone())
    ax[1, 0] = plt.imshow(affinine_x[0, 0, 50], cmap=plt.bone())
    ax[1, 1] = plt.imshow(f_and_a_x[0, 0, 50], cmap=plt.bone())
    """

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, sharex=True, figsize=(12, 6))
    ax1.imshow(x[0, 0, 50], cmap=plt.bone())
    ax2.imshow(flipped_x[0, 0, 50], cmap=plt.bone())
    ax3.imshow(affinine_x[0, 0, 50], cmap=plt.bone())
    ax4.imshow(f_and_a_x[0, 0, 50], cmap=plt.bone())

    plt.savefig(f'imgs/augmentations_{img_cnt}.png')

    if img_cnt == 15:
        exit()

    print(f'img: {img_cnt}')
    img_cnt += 1