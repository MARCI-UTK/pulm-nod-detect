import os
import torch 
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from src.model.data import CropDataset
from src.model.rpn_loss import RegLoss, ClsLoss

from rpn import RPN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataPath = '/data/marci/dlewis37/luna16/'

def main(): 
    img_paths = [os.path.join(dataPath, 'dataset', f) for f in os.listdir(os.path.join(dataPath, 'dataset'))]
    label_paths = [os.path.join(dataPath, 'rpn_labels', f) for f in os.listdir(os.path.join(dataPath, 'rpn_labels'))]
    
    dataset = CropDataset(img_paths=img_paths, label_paths=label_paths)
    batch_size = 32

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size, 
        shuffle=True
    )

    model = RPN(128, 512, 3)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    model.to(f'cuda:{model.device_ids[0]}')

    optimizer = optim.Adam(model.parameters(), lr=0.0001 )
    reg_loss_func = RegLoss()
    cls_loss_func = ClsLoss()

    epochs = 15
    for e in range(epochs): 

        total_epoch_loss = 0
        with tqdm(train_loader) as pbar:
            for idx, data in enumerate(pbar):
                fname, x, y, bb_y = data

                # Make labels correct size 
                y = y.unsqueeze(1)

                x = x.to(f'cuda:{model.device_ids[0]}')
                y = y.to(f'cuda:{model.device_ids[0]}')
                bb_y = bb_y.to(f'cuda:{model.device_ids[0]}')
                            
                optimizer.zero_grad()

                pred_anch_locs, pred_cls_scores = model(x)
    
                cls_loss = cls_loss_func(pred_cls_scores, y, batch_size)
                bb_loss = reg_loss_func(y, pred_anch_locs, bb_y, batch_size)

                loss = cls_loss + bb_loss
                loss.backward()

                optimizer.step()

                total_epoch_loss += loss
                pbar.set_postfix(loss=loss.item())

        avg_epoch_loss = total_epoch_loss / len(train_loader)
       
        print(f'finished epoch {e}. Avg. Loss: {avg_epoch_loss}')
    
    return 

if __name__ == "__main__": 
    main()
