import os 
import torch
from torch.utils.data import DataLoader
from src.model.data import CropDataset
from src.model.bottleneckBlock3d import Bottleneck3d

dataPath = '/data/marci/dlewis37/luna16/'

def main(): 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    paths = [os.path.join(dataPath, 'dataset', f) for f in os.listdir(os.path.join(dataPath, 'dataset'))]
    
    dataset = CropDataset(paths=paths)

    dataLoader = DataLoader(
        dataset=dataset,
        batch_size=5, 
        shuffle=True
    )

    model = Bottleneck3d(1, 64, 4, True, 2)
    model.to(device)
    print('model on gpu.')

    for idx, data in enumerate(dataLoader): 
        x, y = data

        x = x.to(device)
        y = y.to(device)
        print('x and y on gpu.')

        out = model(x)
        print(f'model output shape: {out.shape}')

        break 

if __name__ == '__main__': 
    main()

# Expected output: Expected output "torch.Size([1, 256, 56, 56])"