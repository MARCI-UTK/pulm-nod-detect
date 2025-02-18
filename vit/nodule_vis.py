import os
import util
import matplotlib.pyplot as plt
from dataset import ScanDataset
from torch.utils.data import DataLoader

data_path = '/data/marci/luna16/'
annotations_path = os.path.join(data_path, 'csv', 'annotations.csv')

dataset = ScanDataset(root_path=data_path)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

count = 0
for idx, d in enumerate(dataloader): 
    img, lbl, locs, id = d

    if lbl == 1: 
        print(f'id: {id}')
        print(f'locs: {locs}')
        count += 1
    
    if count > 4: 
        exit()
    """
    locs = util.xyzd_2_2corners(locs[0])

    c1, c2 = locs 

    x1 = int(c1[0].item())
    x2 = int(c2[0].item())

    y1 = int(c1[1].item())
    y2 = int(c2[1].item())

    z1 = int(c1[2].item())
    z2 = int(c2[2].item())

    nodule = img[:, :, x1:x2 + 1, y1:y2 + 1, z1:z2 + 1]

    for i in range(nodule.shape[4]): 
        plt.imshow(nodule[0, 0, :, :, i], cmap=plt.bone())
        plt.savefig(f'nod_{i}.png')
        plt.cla()

    break 
    """