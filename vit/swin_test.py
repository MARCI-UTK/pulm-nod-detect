import torch
import torch.version
from tqdm import tqdm
from src.util.util import makeDataLoaders
from swin_transformer import *

batch_size = 32
train_loader, val_loader = makeDataLoaders(batch_size=batch_size) 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

swin_embed = PatchEmbed().cuda()
swin_t = SwinTransformerBlock(dim=1, input_resolution=(12,12,12), num_heads=1).cuda()

print(torch.version.cuda)
exit()

with tqdm(train_loader) as pbar:
    for idx, data in enumerate(pbar):
        fname, x, y, bb_y = data

        x = x.to(device)

        patches = swin_embed(x)

        print(patches.shape)
        out = swin_t(patches)
        print(out.shape)

        break 