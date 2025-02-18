import os
import util
import torch
import einops
import torchvision
import numpy as np 
from tqdm import tqdm
from unetr import UNETR
from torch.optim import AdamW
from scipy.ndimage import zoom
import matplotlib.pyplot as plt 
from dataset import ScanDataset
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits

data_path = '/data/marci/luna16/'
dataset = ScanDataset(root_path=data_path)
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

model = UNETR(img_shape=(128, 128, 128), input_dim=3, output_dim=1, patch_size=16, embed_dim=768, dropout=0)
model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
model.to(f'cuda:{model.device_ids[0]}')

optimizer = AdamW(params=model.parameters(), lr=0.0001)

dice_loss_func = util.DiceLoss()

epochs = 10
losses = []
accs = []
for e in range(epochs): 
    with tqdm(dataloader) as pbar: 
        train_loss = 0
        val_loss = 0

        model.train()
        for idx, data in enumerate(pbar): 
            optimizer.zero_grad()

            x, mask, loc, y = data
 
            x = x.to(f'cuda:{model.device_ids[0]}')
            mask = mask.to(f'cuda:{model.device_ids[0]}')
            out = model(x)
            
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(out, mask, reduction='mean')
            focal_loss = (torchvision.ops.sigmoid_focal_loss(out, mask, alpha=0.25, gamma=3, reduction='mean'))
            dice_loss = dice_loss_func(out, mask)

            loss = 100 * focal_loss

            loss.backward()
            optimizer.step()

            if (e % 2 == 0) and (idx % 20 == 0): 
                for i in range(4): 
                    if y[i] == 1: 
                        util.visualize_attention_map(x[i][0].detach().cpu(), mask[i][0].detach().cpu(), out[i][0].detach().cpu(), loc[i], e, idx)
                        break

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            losses.append(loss.item())
        
        print(f'epoch training loss: {train_loss / len(dataloader)}')
    
print(losses)
print(accs)