import os
import torch 
from torch.utils.data import DataLoader
from torch.nn.modules.loss import BCELoss
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from src.model.data import CropDataset
from src.model.model import TestModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataPath = '/data/marci/dlewis37/luna16/'

def main(): 
    paths = [os.path.join(dataPath, 'dataset', f) for f in os.listdir(os.path.join(dataPath, 'dataset'))]
    
    dataset = CropDataset(paths=paths)

    dataLoader = DataLoader(
        dataset=dataset,
        batch_size=5, 
        shuffle=True
    )

    model = TestModel()
    model.to(device)
    lossFunc = BCELoss()
    optimizer = optim.Adam(model.parameters())

    epochs = 20
    losses = []
    for e in range(epochs): 

        totalEpochLoss = 0
        for idx, data in enumerate(dataLoader): 
            x, y = data
            y = y.unsqueeze(1)

            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()

            output = model(x)
            
            loss = lossFunc(output, y)

            loss.backward()
            optimizer.step()

            totalEpochLoss += loss.item()
        
        avgEpochLoss = totalEpochLoss / len(dataLoader)
        losses.append(avgEpochLoss)

        print(f'finished epoch {e}. Loss: {avgEpochLoss}')


    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.savefig('graphs/loss_per_epoch.png')

    return 

if __name__ == "__main__": 
    main()