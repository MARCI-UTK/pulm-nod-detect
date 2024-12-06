import os
import torch 
from vit import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from dataset import CropDataset
from torch.utils.data import DataLoader
from clearml import Task
import matplotlib.pyplot as plt
import math
import numpy as np

def calculate_metrics(logits, metrics):
    probs = torch.sigmoid(logits)
    pred_binary = torch.where(probs >= 0.5, 1, 0)
    
    tp = ((pred_binary == 1.) & (y == 1.)).sum().item()
    tn = ((pred_binary == 0.) & (y == 0.)).sum().item()
    fp = ((pred_binary == 1.) & (y == 0.)).sum().item()
    fn = ((pred_binary == 0.) & (y == 1.)).sum().item()

    acc = (tp + tn) / (tp + tn + fp + fn)
    
    if tp + fp == 0: 
        pre = 0
    else: 
        pre = tp / (tp + fp)

    if tp + fn == 0: 
        rec = 0
    else: 
        rec = tp / (tp + fn)
    
    if pre + rec == 0: 
        f1  = 0
    else: 
        f1  = (2 * pre * rec) / (pre + rec)

    new_metrics = [acc, pre, rec, f1]
    metrics = [x + y for x, y in zip(metrics, new_metrics)]

    return (acc, metrics)
    
task = Task.init(project_name="Pulmonary Nodule Detection", task_name="1st ViT Training")
logger = task.get_logger()

config = {
        "patch_size": 12,
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "intermediate_size": 4048,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "image_size": 96,
        "num_classes": 1,
        "num_channels": 1,
        "qkv_bias": True,
    }

dataPath = '/data/marci/luna16/'
img_paths = [os.path.join(dataPath, 'crops', f) for f in os.listdir(os.path.join(dataPath, 'crops'))]
label_paths = [os.path.join(dataPath, 'labels', f) for f in os.listdir(os.path.join(dataPath, 'labels'))]

dataset = CropDataset(img_paths=img_paths, label_paths=label_paths)

batch_size = 32
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# NOTE: 7176 positive crops and 599 negative 

model = ViT(config).cuda()
model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
model.to(f'cuda:{model.device_ids[0]}')
optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)

for e in range(10):
    with tqdm(dataloader) as pbar:
        
        train_loss = 0
        train_accuracy = 0
        metrics = [0, 0, 0, 0]

        model.train()
        for idx, data in enumerate(pbar):
            optimizer.zero_grad()

            fname, x, y = data

            x = x.to(model.device_ids[0])

            y = y.to(model.device_ids[0])
            y = y.unsqueeze(1).float()

            logits, attention_maps = model(x)

            """
            # Concatenate the attention maps from all blocks
            attention_maps = torch.cat(attention_maps, dim=1)
            # select only the attention maps of the CLS token
            attention_maps = attention_maps[:, :, 0, 1:]
            # Then average the attention maps of the CLS token over all the heads
            attention_maps = attention_maps.mean(dim=1)
            # Reshape the attention maps to a square
            num_patches = attention_maps.size(-1)
            size = int(math.cbrt(num_patches))
            attention_maps = attention_maps.view(-1, size, size, size)
            # Resize the map to the size of the image
            attention_maps = attention_maps.unsqueeze(1)
            attention_maps = nn.functional.interpolate(attention_maps, size=(96, 96, 96), mode='trilinear', align_corners=False)
            attention_maps = attention_maps.squeeze(1)
            # Plot the images and the attention maps
            fig = plt.figure(figsize=(20, 10))
            mask = np.concatenate([np.ones((96, 96, 96)), np.zeros((96, 96, 96))], axis=1)

            img = x[1][0][x.shape[2] // 2].cpu().numpy()
            plt.imshow(img, cmap=plt.bone())
            # Mask out the attention map of the left image
            extended_attention_map = np.concatenate((np.zeros((96, 96, 96)), attention_maps[1].detach().cpu().numpy()), axis=1)
            extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
            plt.imshow(extended_attention_map[extended_attention_map.shape[0] // 2], alpha=0.5, cmap='jet')
            # Show the ground truth and the prediction
            plt.savefig('test_att_map.png')
            """

            pos_weight = (y == 0).sum() / (y == 1).sum()
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction='mean')

            loss.backward()
            optimizer.step()

            batch_acc, metrics = calculate_metrics(logits=logits, metrics=metrics)

            train_loss += loss.item()

            pbar.set_postfix(loss=loss.item(), acc=batch_acc)

    epoch_train_loss = train_loss / len(dataloader)
    epoch_metrics = [x / len(dataloader) for x in metrics]

    print(f'finished epoch {e}. loss: {epoch_train_loss}. accuracy: {epoch_metrics[0]}')
    print(f'precision: {epoch_metrics[1]}. recall: {epoch_metrics[2]}. f1: {epoch_metrics[3]}.')

    logger.report_scalar(
        "ViT Epoch Loss", " Train Loss", iteration=e, value=epoch_train_loss
    )

    logger.report_scalar(
        "ViT Epoch Accuracy", " Train Accuracy", iteration=e, value=epoch_metrics[0]
    )