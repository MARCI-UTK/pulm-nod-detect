import json
import numpy as np
import torch
import matplotlib.pyplot as plt 
from scipy.ndimage import zoom 
from vit import ViT
import pandas as pd 
from dataset import ScanDataset


config = {
        "patch_size": 16,
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "image_size": 256,
        "num_classes": 1,
        "num_channels": 1,
        "qkv_bias": True,
}

img = '/data/marci/luna16/processed_scan/1.3.6.1.4.1.14519.5.2.1.6279.6001.249530219848512542668813996730.npy'

img = np.load(img)

z, y, x = img.shape
img = zoom(img, (256 / z, 256 / y, 256 / x))
img = (img - np.min(img)) / (np.max(img) - np.min(img)) 

img = torch.Tensor(img).float().to('cuda:2')
img = img.unsqueeze(0).unsqueeze(0)
print(img.shape)

model = ViT(config).to('cuda:2')
logits, attention = model(img)
print(logits)