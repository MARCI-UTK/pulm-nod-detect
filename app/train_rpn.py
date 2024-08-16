import os
import torch 
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from src.model.rpn_loss import RegLoss, ClsLoss, ValClsLoss
from src.util.util import *

from rpn import RPN, get_centers, get_anc_boxes
from roi import ROI, CropProposals

import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
from clearml import Task, Logger

task = Task.init(project_name="Pulmonary Nodule Detection", task_name="Nature Paper Loss Function (0.12 pos. threshold)")
logger = task.get_logger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataPath = '/data/marci/dlewis37/luna16/'
batch_size = 32
    
def main(): 
    # This function needs to be better (take parameters)
    train_loader, val_loader = makeDataLoaders() 

    # Get centers for anchor boxes
    centers_list = get_centers(0, 0)

    # Get list of anchor box coordinates in [x, y, z, d] form 
    anc_box_list = get_anc_boxes(centers=centers_list)
    anc_box_list = [corners_2_xyzd(x) for x in anc_box_list]
    anc_box_list = torch.tensor(anc_box_list)

    # Initialize model 
    rpn = RPN(128, 512, 3)
    rpn.apply(weight_init)
    rpn = torch.nn.DataParallel(rpn, device_ids=[0,1,2,3])
    rpn.to(f'cuda:{rpn.device_ids[0]}')

    """
    roi = ROI()
    roi.apply(weight_init)
    roi = torch.nn.DataParallel(roi, device_ids=[0,1,2,3])
    roi.to(f'cuda:{roi.device_ids[0]}')

    proposal_fm_generator = CropProposals()
    """

    # Create optimizer and LR scheduler 
    optimizer = optim.Adam(rpn.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # Define loss functions 
    reg_loss_func = RegLoss()
    cls_loss_func = ClsLoss()
    val_cls_loss_func = ValClsLoss()

    train_losses = []   
    val_losses = []

    epochs = 50
    for e in range(epochs): 

        train_loss = 0
        rpn_cm = [0, 0, 0, 0]
        roi_cm = [0, 0, 0, 0]
        
        rpn.train()
        with tqdm(train_loader) as pbar:
            for idx, data in enumerate(pbar):
                fname, x, y, bb_y = data

                # This is used for logging 
                fname = list(fname)

                # Make labels correct size 
                y = y.unsqueeze(1)

                x = x.to(f'cuda:{rpn.device_ids[0]}')
                y = y.to(f'cuda:{rpn.device_ids[0]}')
                bb_y = bb_y.to(f'cuda:{rpn.device_ids[0]}')
                            
                optimizer.zero_grad()

                pred_anch_locs, pred_cls_scores, fm = rpn(x)
                update_cm(y, pred_cls_scores, rpn_cm)

                if e == 1: 
                    fm = fm.detach().tolist()
                    plt.imshow(fm[len(fm) // 2][len(fm[0]) // 2][12], cmap='gray') 
                    plt.savefig('feature_map.png')
                    plt.show()
                
                sampled_pred, sampled_target = sample_anchor_boxes(pred_cls_scores, y) 

                pred_binary = torch.where(sampled_pred > 0.12, 1., 0.)

                cls_loss = cls_loss_func(pred_binary, sampled_target, 32)
                reg_loss = reg_loss_func(y, pred_anch_locs, bb_y, 32)

                cls_loss.requires_grad = True
                #reg_loss.requires_grad = True

                loss = cls_loss + reg_loss
                print(loss.requires_grad)
                loss.backward()

                train_loss += loss.item()

                iteration = e * len(train_loader) + idx
                logger.report_scalar(
                    "Cls. vs. Reg. Loss", "Reg. Loss", iteration=iteration, value=reg_loss,
                )
                logger.report_scalar(
                    "Cls. vs. Reg. Loss", "Cls. Loss", iteration=iteration, value=cls_loss,
                )
                logger.report_scalar(
                    "Total Loss", "Total Loss", iteration=iteration, value=loss.item(),
                )

                # Clip gradients and update parameters  
                torch.nn.utils.clip_grad_norm_(rpn.parameters(), 1.0)

                optimizer.step() 
                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
                
        if e > 25: 
            scheduler.step()

        # Take average of losses using the number of batches
        epoch_train_loss = train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Print epoch losses and confusion matrix statistics
        print(f'finished epoch {e}. Train Loss: {epoch_train_loss}.')
        print(f'rpn [tp, tn, fp, fn]: [{rpn_cm[0]}, {rpn_cm[1]}, {rpn_cm[2]}, {rpn_cm[3]}].')

    print(f'train losses: {train_losses}.')

    #writer.close()

    return 

if __name__ == "__main__": 
    main()