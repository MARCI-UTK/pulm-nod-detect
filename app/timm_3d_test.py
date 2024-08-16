import timm_3d
import torch 
from src.util.util import makeDataLoaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataPath = '/data/marci/dlewis37/luna16/'
batch_size = 32
    
train_loader, val_loader = makeDataLoaders() 

m = timm_3d.create_model(
    'resnet101.a1_in1k',
    pretrained=True,
    num_classes=0,
    global_pool=''
)

m.to(device)

# Shape of input (B, C, H, W, D). B - batch size, C - channels, H - height, W - width, D - depth

for data in train_loader: 
    fname, x, y, bb_y = data

    channel_expansion_x = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3], x.shape[4])
    
    for i in range(len(x)): 
        channel_expansion_x[i, 0] = x[i][0]
        channel_expansion_x[i, 1] = x[i][0]
        channel_expansion_x[i, 2] = x[i][0]

    channel_expansion_x = channel_expansion_x.to(device)

    res = m(channel_expansion_x)
    print(res.shape)

    exit()