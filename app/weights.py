from src.util.util import makeDataLoaders

def get_pos_weight_val(y): 
    pos_weight = 0

    n_pos = (y == 1.).sum()
    n_neg = (y == 0.).sum()

    if n_pos == 0: 
        pos_weight = n_neg
    else: 
        pos_weight = n_neg / n_pos

    norm = (pos_weight * n_pos) + n_neg

    w_p = pos_weight / norm
    w_n = 1 / norm

    return w_p, w_n

import torch

def criterion(pred, target): 
    w_p, w_n = get_pos_weight_val(target)

    pred = torch.clamp(pred, min=1e-8, max=1-1e-8)  
    loss =  w_p * (target * torch.log(pred)) + (w_n * ((1 - target) * torch.log(1 - pred)))

    return loss

if __name__ == "__main__": 
    dataPath = '/data/marci/luna16/'
    train_loader, val_loader = makeDataLoaders()

    for d in train_loader: 
        _, _, y, _ = d

        # Make labels correct size 
        y = y.unsqueeze(1)
        y = y.to(f'cuda:0')

        w_p, w_n = get_pos_weight_val(y)
