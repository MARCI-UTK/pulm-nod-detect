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

    train_img_idxs = int(len(img_paths) * 0.8)
    train_img_paths = img_paths[:train_img_idxs - 1]
    val_img_paths   = img_paths[train_img_idxs:]

    train_label_idxs = int(len(img_paths) * 0.8)
    train_label_paths = label_paths[:train_label_idxs - 1]
    val_label_paths   = label_paths[train_label_idxs:]

    dataset = CropDataset(img_paths=img_paths, label_paths=label_paths)
    train_data = CropDataset(img_paths=train_img_paths, label_paths=train_label_paths)
    val_data   = CropDataset(img_paths=val_img_paths, label_paths=val_label_paths)
    batch_size = 32

    dataset = CropDataset(img_paths=img_paths, label_paths=label_paths)
    batch_size = 32

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size, 
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size, 
        shuffle=True
    )

    model = RPN(128, 512, 3)
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    model.to(f'cuda:{model.device_ids[0]}')

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    reg_loss_func = RegLoss()
    cls_loss_func = ClsLoss()

    train_losses = []
    val_losses = []

    epochs = 25
    for e in range(epochs): 

        train_loss = 0
        avg_cls_loss = 0
        avg_reg_loss = 0
        val_loss = 0
        
        model.train()
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

                train_loss += loss.item()
                avg_cls_loss += cls_loss
                avg_reg_loss += bb_loss
                pbar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                fname, x, y, bb_y = data

                y = y.unsqueeze(1)

                x = x.to(f'cuda:{model.device_ids[0]}')
                y = y.to(f'cuda:{model.device_ids[0]}')
                bb_y = bb_y.to(f'cuda:{model.device_ids[0]}')

                pred_anch_locs, pred_cls_scores = model(x)
    
                cls_loss = cls_loss_func(pred_cls_scores, y, batch_size)
                bb_loss = reg_loss_func(y, pred_anch_locs, bb_y, batch_size)

                loss = cls_loss + bb_loss 
                val_loss += loss.item()

        epoch_train_loss = train_loss / len(train_loader)
        #avg_reg_loss /= len(train_loader)
        #avg_cls_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)

        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(f'finished epoch {e}. Train Loss: {epoch_train_loss}. Val Loss: {epoch_val_loss}')
        #print(f'avg cls loss: {avg_cls_loss}. avg reg loss: {avg_reg_loss}.')
    
    print(f'train losses: {train_losses}.')
    print(f'val losses: {val_losses}.')
    return 

if __name__ == "__main__": 
    main()
