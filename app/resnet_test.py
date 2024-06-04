import os 
import torch
import torch.optim as optim
from torch.nn.modules.loss import BCELoss
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.model.data import CropDataset
from src.model.bottleneckBlock3d import Bottleneck3d
from src.model.resNet3d import ResNet3d

dataPath = '/data/marci/dlewis37/luna16/'

def main(): 

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    paths = [os.path.join(dataPath, 'dataset', f) for f in os.listdir(os.path.join(dataPath, 'dataset'))]
    
    dataset = CropDataset(paths=paths)

    dataLoader = DataLoader(
        dataset=dataset,
        batch_size=5, 
        shuffle=True
    )

    # resnetX = (Num of channels, repetition, Bottleneck_expansion , Bottleneck_layer)
    model_parameters = {}
    model_parameters['resnet50']  = ([64, 128, 256, 512], [3, 4, 6, 3], 4, True)
    model_parameters['resnet101'] = ([64,128,256,512],[3,4,23,3],4,True)
    model_parameters['resnet152'] = ([64,128,256,512],[3,8,36,3],4,True)

    architecture = 'resnet50'
    model = ResNet3d(model_parameters[architecture], in_channels=1, num_classes=1)
    # model = Bottleneck3d(in_channels=1, intermediate_channels=6, expansion=4, is_Bottleneck=True, stride=1)
    model.to(device)

    lossFunc = BCELoss()
    optimizer = optim.Adam(model.parameters())
    epochs = 1

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

            print(loss.item())
            break

            totalEpochLoss += loss.item()
        
        avgEpochLoss = totalEpochLoss / len(dataLoader)
        losses.append(avgEpochLoss)

        print(f'finished epoch {e}. Loss: {avgEpochLoss}')

    """
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.savefig('graphs/loss_per_epoch.png')
    """

    return 

if __name__ == "__main__": 
    main()

# Expected output: Expected output "torch.Size([1, 256, 56, 56])"