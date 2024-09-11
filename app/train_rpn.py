import os
import torch 
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.model.rpn_loss import RegLoss, ClsLoss, ValClsLoss, NatureClsLoss
from src.util.util import *

from rpn import RPN, get_centers, get_anc_boxes
from roi import ROI, CropProposals
from src.model.feature_extractor import FeatureExtractor

import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
from clearml import Task, Logger

from sklearn.metrics import roc_curve

task = Task.init(project_name="Pulmonary Nodule Detection", task_name="RPN Big Augmented Data + Momentum + L2 (0.0001) + LR Scheduler (init = 0.01) + Hopfield")
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
    # CHANGE DEV IDS BACK TO INCLUDE 0!!!

    fe = FeatureExtractor()
    fe.apply(weight_init)
    fe = torch.nn.DataParallel(fe, device_ids=[1,2,3])
    fe.to(f'cuda:{fe.device_ids[0]}')

    rpn = RPN(128, 512, 3)
    rpn = torch.nn.DataParallel(rpn, device_ids=[1,2,3])
    rpn.to(f'cuda:{rpn.device_ids[0]}')

    # Create optimizer and LR scheduler 
    optimizer = optim.SGD(list(fe.parameters()) + list(rpn.parameters()), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # Define loss functions 
    reg_loss_func = RegLoss()
    cls_loss_func = ClsLoss()

    train_losses = []   
    
    epochs = 100
    for e in range(epochs): 

        train_loss = 0
        val_loss = 0
        train_cm = [0, 0, 0, 0]
        val_cm   = [0, 0, 0, 0]
        
        fe.train()
        rpn.train()
        with tqdm(train_loader) as pbar:
            for idx, data in enumerate(pbar):
                fname, x, y, bb_y = data

                # This is used for logging 
                fname = list(fname)

                # Make labels correct size 
                y = y.unsqueeze(1)

                x = x.to(f'cuda:{fe.device_ids[0]}')

                y = y.to(f'cuda:{rpn.device_ids[0]}')
                bb_y = bb_y.to(f'cuda:{rpn.device_ids[0]}')
                            
                optimizer.zero_grad()

                fm = fe(x)
                pred_anch_locs, pred_cls_scores = rpn(fm)

                update_cm(y, pred_cls_scores, train_cm)
                
                sampled_pred, sampled_target = sample_anchor_boxes(pred_cls_scores, y) 

                cls_loss = cls_loss_func(sampled_pred, sampled_target)
                reg_loss = reg_loss_func(y, pred_anch_locs, bb_y)

                loss = cls_loss + reg_loss
                loss.backward()

                train_loss += loss.item()

                # Clip gradients and update parameters  
                torch.nn.utils.clip_grad_norm_(fe.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(rpn.parameters(), 1.0)
                optimizer.step() 

                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
            
        fe.eval()
        rpn.eval()
        for idx, data in enumerate(val_loader):
                fname, x, y, bb_y = data

                # This is used for logging 
                fname = list(fname)

                # Make labels correct size 
                y = y.unsqueeze(1)

                x = x.to(f'cuda:{fe.device_ids[0]}')
                y = y.to(f'cuda:{rpn.device_ids[0]}')
                bb_y = bb_y.to(f'cuda:{rpn.device_ids[0]}')
                            
                fm = fe(x)
                pred_anch_locs, pred_cls_scores = rpn(fm)

                update_cm(y, pred_cls_scores, val_cm)

                sampled_pred, sampled_target = sample_anchor_boxes(pred_cls_scores, y) 

                cls_loss = cls_loss_func(sampled_pred, sampled_target)
                reg_loss = reg_loss_func(y, pred_anch_locs, bb_y)

                loss = cls_loss + reg_loss

                val_loss += loss.item()

        # Take average of losses using the number of batches
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)

        train_acc = (train_cm[0] + train_cm[1]) / sum(train_cm)
        val_acc = (val_cm[0] + val_cm[1]) / sum(val_cm)

        train_sens = (train_cm[0] / (train_cm[0] + train_cm[3]))
        val_sens = (val_cm[0] / (val_cm[0] + val_cm[3]))


        logger.report_scalar(
            "Epoch Total Loss", "Train Loss", iteration=e, value=epoch_train_loss
        )
        logger.report_scalar(
            "Epoch Total Loss", "Val. Loss", iteration=e, value=epoch_val_loss
        )
        logger.report_scalar(
            "Epoch Accuracy", "Train Acc.", iteration=e, value=train_acc
        )
        logger.report_scalar(
            "Epoch Accuracy", "Val. Acc.", iteration=e, value=val_acc
        )
        logger.report_scalar(
            "Epoch Sensitivity", "Train Sens.", iteration=e, value=train_sens
        )
        logger.report_scalar(
            "Epoch Sensitivity", "Val. Sens.", iteration=e, value=val_sens
        )

        # Print epoch losses and confusion matrix statistics
        print(f'finished epoch {e}. Train Loss: {epoch_train_loss}. Val Loss: {epoch_val_loss}')
        print(f'Train Acc.: {train_acc}. Val Acc.: {val_acc}') 
        print(f'train [tp, tn, fp, fn]: [{train_cm[0]}, {train_cm[1]}, {train_cm[2]}, {train_cm[3]}].')
        print(f'val   [tp, tn, fp, fn]: [{val_cm[0]}, {val_cm[1]}, {val_cm[2]}, {val_cm[3]}].')

        scheduler.step()

    return 

if __name__ == "__main__": 
    main()