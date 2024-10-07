import os
import time
import torch 
import torch.optim as optim
import torch.nn.functional as F 
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.util.util import *

from rpn import RPN, get_centers, get_anc_boxes, rpn_iteration
from roi import ROI, CropProposals, roi_iteration
from src.model.feature_extractor import FeatureExtractor

import matplotlib.pyplot as plt
from clearml import Task
from sklearn.metrics import roc_curve

task = Task.init(project_name="Pulmonary Nodule Detection", task_name="ROI 2 Optimizers")
logger = task.get_logger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataPath = '/data/marci/luna16/'
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
    # CHANGE DEV IDS BACK TO INCLUDE 0!!!

    fe = FeatureExtractor()
    fe.apply(weight_init)
    fe = torch.nn.DataParallel(fe, device_ids=[0,1,2,3])
    fe.to(f'cuda:{fe.device_ids[0]}')

    rpn = RPN(128, 512, 3)
    rpn = torch.nn.DataParallel(rpn, device_ids=[0,1,2,3])
    rpn.to(f'cuda:{rpn.device_ids[0]}')

    roi = ROI()
    roi.apply(weight_init)
    roi = torch.nn.DataParallel(roi, device_ids=[0,1,2,3])
    roi.to(f'cuda:{roi.device_ids[0]}')

    crp = CropProposals()
    crp = torch.nn.DataParallel(crp, device_ids=[0,1,2,3])
    crp.to(f'cuda:{crp.device_ids[0]}')

    # Create optimizer and LR scheduler 
    rpn_optimizer = optim.SGD(list(fe.parameters()) + list(rpn.parameters()), lr=0.01, weight_decay=0.0001, momentum=0.9)
    roi_optimizer = optim.SGD(list(roi.parameters()), lr=0.01, weight_decay=0.0001, momentum=0.9)
    
    rpn_scheduler = optim.lr_scheduler.StepLR(rpn_optimizer, step_size=50, gamma=0.1)
    roi_scheduler = optim.lr_scheduler.StepLR(roi_optimizer, step_size=50, gamma=0.1)
    
    anc_box_list = anc_box_list.to(f'cuda:{rpn.device_ids[0]}')

    #torch.autograd.set_detect_anomaly(True)

    epochs = 100
    for e in range(epochs): 

        rpn_train_loss = 0
        roi_train_loss = 0
        rpn_val_loss   = 0
        roi_val_loss   = 0
        train_cm = [0, 0, 0, 0]
        val_cm   = [0, 0, 0, 0]
        
        fe.train()
        rpn.train()
        roi.train()
        with tqdm(train_loader) as pbar:
            for idx, data in enumerate(pbar):
                rpn_optimizer.zero_grad()
                roi_optimizer.zero_grad()

                # Get region proposals 
                rpn_loss, fm, cls_scores, anc_locs = rpn_iteration(data, fe, rpn)

                # Backprop and update parameters for RPN
                rpn_loss.backward()
                torch.nn.utils.clip_grad_norm_(fe.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(rpn.parameters(), 1.0)
                rpn_optimizer.step()

                # Detach tensors needed for ROI operations from computation graph of RPN 
                _, _, y, bb_y = data
                y = y.detach().requires_grad_(True)
                bb_y = bb_y.detach().requires_grad_(True)
                fm = fm.detach().requires_grad_(True)
                cls_scores = cls_scores.detach().requires_grad_(True)
                anc_locs = anc_locs.detach().requires_grad_(True)

                anc_box_list = anc_box_list.to(f'cuda:{roi.device_ids[0]}')
                
                # Classify output of RPN using ROI
                roi_loss, cls_scores, y = roi_iteration(y, bb_y, fm, anc_box_list, roi, crp, 
                                                        cls_scores, anc_locs, 2500)

                update_cm(y, cls_scores, train_cm)
                
                # Backprop and update parameters for ROI
                roi_loss.backward()
                torch.nn.utils.clip_grad_norm_(roi.parameters(), 1.0)
                roi_optimizer.step()
                
                rpn_train_loss += rpn_loss.item()
                roi_train_loss += roi_loss.item()

                pbar.set_postfix(rpn_loss=rpn_loss.item(), roi_loss=roi_loss.item(), lr=rpn_optimizer.param_groups[0]['lr'])

        fe.eval()
        rpn.eval()
        roi.eval()
        for idx, data in enumerate(val_loader):
            # Get region proposals 
            rpn_loss, fm, cls_scores, anc_locs = rpn_iteration(data, fe, rpn)

            anc_box_list = anc_box_list.to(f'cuda:{roi.device_ids[0]}')
            _, _, y, bb_y = data
             
            # Classify output of RPN using ROI
            roi_loss = roi_iteration(y, bb_y, fm, anc_box_list, roi, crp, 
                                     cls_scores, anc_locs, 2500)
         
            # Classify output of RPN using ROI
            roi_loss, cls_scores, y = roi_iteration(y, bb_y, fm, anc_box_list, roi, crp, 
                                                    cls_scores, anc_locs, 2500)

            update_cm(y, cls_scores, val_cm)

            rpn_val_loss += rpn_loss.item()
            roi_val_loss += roi_loss.item() 

        # Take average of losses using the number of batches
        epoch_rpn_train_loss = rpn_train_loss / len(train_loader)
        epoch_roi_train_loss = roi_train_loss / len(train_loader)
        epoch_rpn_val_loss = rpn_val_loss / len(val_loader)
        epoch_roi_val_loss = roi_val_loss / len(val_loader)

        logger.report_scalar(
            "Epoch RPN Loss", " Train Loss", iteration=e, value=epoch_rpn_train_loss
        )
        logger.report_scalar(
            "Epoch RPN Loss", "Val. Loss", iteration=e, value=epoch_rpn_val_loss
        )
        logger.report_scalar(
            "Epoch ROI Loss", "Train Loss", iteration=e, value=epoch_roi_train_loss
        )
        logger.report_scalar(
            "Epoch ROI Loss", "Val. Loss", iteration=e, value=epoch_roi_val_loss
        )
      
        # Print epoch losses and confusion matrix statistics
        print(f'finished epoch {e}. RPN Train Loss: {epoch_rpn_train_loss}. ROI Train Loss: {epoch_roi_train_loss}.')
        print(f'finished epoch {e}. RPN Val. Loss: {epoch_rpn_val_loss}. ROI Val. Loss: {epoch_roi_val_loss}.')
        print(f'train [tp, tn, fp, fn]: [{train_cm[0]}, {train_cm[1]}, {train_cm[2]}, {train_cm[3]}].')
        print(f'val   [tp, tn, fp, fn]: [{val_cm[0]}, {val_cm[1]}, {val_cm[2]}, {val_cm[3]}].')

        rpn_scheduler.step()
        roi_scheduler.step()

    return 

if __name__ == "__main__": 
    main()