import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

files = '/data/marci/dlewis37/msd/Task06_Lung/imagesTr/' 
train_imgs = [os.path.join(files, f) for f in os.listdir(files) if not f.startswith('.')]

labels = '/data/marci/dlewis37/msd/Task06_Lung/labelsTr/' 
train_labels = [os.path.join(labels, f) for f in os.listdir(labels) if not f.startswith('.')]

img = nib.load(train_labels[0])
img = np.array(img.dataobj)
img = np.transpose(img, (1, 0, 2))

plt.imshow(img[:, :, img.shape[2] // 4], cmap=plt.bone())
plt.savefig('new_data.png')