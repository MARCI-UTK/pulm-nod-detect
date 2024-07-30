import os
import torch 
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from src.model.rpn_loss import RegLoss, ClsLoss, ValClsLoss
from src.util.util import *

from rpn import RPN, get_centers, get_anc_boxes
from roi import ROI, CropProposals

import torch.nn.functional as F

#from clearml import Task
#from torch.utils.tensorboard import SummaryWriter 

#task = Task.init(project_name="Pulmonary Nodule Detection", task_name="Loss Logging (75e. Manual L2 regularization.)")
#writer = SummaryWriter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataPath = '/data/marci/dlewis37/luna16/'
batch_size = 128
    
def main(): 
    # This function needs to be better (take parameters)
    train_loader, val_loader = makeROILoader() 
   
    roi = ROI()
    roi.apply(weight_init)
    roi = torch.nn.DataParallel(roi, device_ids=[0,1,2,3])
    roi.to(f'cuda:{roi.device_ids[0]}')

    # Create optimizer and LR scheduler 
    optimizer = optim.Adam(roi.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    train_losses = []   
    val_losses = []

    epochs = 30
    for e in range(epochs): 

        train_loss = 0
        train_cm = [0, 0, 0, 0]
        
        roi.train()
        with tqdm(train_loader) as pbar:
            for idx, data in enumerate(pbar):
                fname, x, y, gt_delta, anc_box = data

                # This is used for logging 
                fname = list(fname)

                # Make labels correct size 
                y = y.unsqueeze(1)

                x = x.to(f'cuda:{roi.device_ids[0]}')
                y = y.to(f'cuda:{roi.device_ids[0]}')
                gt_delta = gt_delta.to(f'cuda:{roi.device_ids[0]}')
                            
                optimizer.zero_grad()

                pred_cls, pred_deltas = roi(x)

                
                cls_loss = F.binary_cross_entropy_with_logits(input=pred_cls, target=y)
                reg_loss = F.smooth_l1_loss(input=pred_deltas, target=gt_delta, beta=1)

                # Calculate loss and backpropogate 
                loss = cls_loss + reg_loss
                loss.backward()

                # Clip gradients and update parameters  
                torch.nn.utils.clip_grad_norm_(roi.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

                # Update confustion matrix for current epoch 
                update_cm(y, pred_cls, train_cm) 

                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

                #writer.add_scalar('Training Loss', loss.item(), e * len(train_loader) + idx)
                
        scheduler.step()

        # Take average of losses using the number of batches
        epoch_train_loss = train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Print epoch losses and confusion matrix statistics
        print(f'finished epoch {e}. Train Loss: {epoch_train_loss}.')
        print(f'train [tp, tn, fp, fn]: [{train_cm[0]}, {train_cm[1]}, {train_cm[2]}, {train_cm[3]}].')

    print(f'train losses: {train_losses}.')

    #writer.close()

    return 

if __name__ == "__main__": 
    main()