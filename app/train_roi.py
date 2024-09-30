import os
import time
import torch 
import torch.optim as optim
import torch.nn.functional as F 
from tqdm import tqdm
import numpy as np
import pandas as pd

from src.model.rpn_loss import RegLoss, ClsLoss, ValClsLoss, NatureClsLoss
from src.util.util import *

from rpn import RPN, get_centers, get_anc_boxes
from roi import ROI, CropProposals
from src.model.feature_extractor import FeatureExtractor

import matplotlib.pyplot as plt
from clearml import Task
from sklearn.metrics import roc_curve

task = Task.init(project_name="Pulmonary Nodule Detection", task_name="Testing ROI")
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
    fe = torch.nn.DataParallel(fe, device_ids=[1,2,3])
    fe.to(f'cuda:{fe.device_ids[0]}')

    rpn = RPN(128, 512, 3)
    rpn = torch.nn.DataParallel(rpn, device_ids=[1,2,3])
    rpn.to(f'cuda:{rpn.device_ids[0]}')

    roi = ROI()
    roi = torch.nn.DataParallel(roi, device_ids=[1,2,3])
    roi.to(f'cuda:{roi.device_ids[0]}')

    crp = CropProposals()
    crp = torch.nn.DataParallel(crp, device_ids=[1,2,3])
    crp.to(f'cuda:{crp.device_ids[0]}')

    # Create optimizer and LR scheduler 
    optimizer = optim.SGD(list(fe.parameters()) + list(rpn.parameters()) + list(roi.parameters()), lr=0.01, weight_decay=0.0001, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    epochs = 100
    for e in range(epochs): 

        train_loss = 0
        val_loss = 0
        train_cm = [0, 0, 0, 0]
        val_cm   = [0, 0, 0, 0]
        
        fe.train()
        rpn.train()
        roi.train()
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

                anc_box_list = anc_box_list.to(f'cuda:{rpn.device_ids[0]}')

                top_n = 1000

                start = time.time()
                corners, mask = rpn_to_roi(cls_scores=pred_cls_scores, pred_locs=pred_anch_locs, 
                                           anc_boxes=anc_box_list, nms_thresh=0.1, top_n=top_n)
                end = time.time()
                #print(f'IOU NMS time: {end - start}.')

                start = time.time()
                proposals = crp(fm, corners)
                end = time.time()
                #print(f'Crop proposals time: {end - start}.')

                pred_cls_scores, pred_anch_locs = roi(proposals)

                top_n_y = []
                top_n_bb_y = []
                final_mask = []
                y = y.squeeze()
                pred_cls_scores = pred_cls_scores.squeeze()
                for i in range(len(y)): 
                    c_i = pred_cls_scores[i]
                    y_i = y[i]
                    b_i = bb_y[i] 
                    m_i = mask[i]
                    _, sorted_idxs = torch.sort(c_i, descending=True)

                    top_n_y.append(y_i[sorted_idxs])
                    top_n_bb_y.append(b_i[sorted_idxs])
                    final_mask.append(m_i[sorted_idxs])

                top_n_y = torch.stack(top_n_y)
                top_n_bb_y = torch.stack(top_n_bb_y)
                final_mask = torch.stack(final_mask)

                if (top_n_y == 1.).sum() == 0: 
                    pos_weight = (top_n_y == 0).sum()
                else: 
                    pos_weight = (top_n_y == 0.).sum() / (top_n_y == 1.).sum()

                cls_loss = F.binary_cross_entropy_with_logits(pred_cls_scores, top_n_y, pos_weight=pos_weight, reduction='none')
                reg_loss = F.smooth_l1_loss(pred_anch_locs, top_n_bb_y, reduction='none', beta=1)

                print((top_n_y != -1).sum())

                final_mask &= (top_n_y != -1)

                reg_mask = final_mask.detach().clone()
                reg_mask &= (top_n_y == 1)

                print(reg_mask.sum(), final_mask.sum())

                reg_mask = reg_mask.float()
                final_mask = final_mask.float()

                cls_loss = final_mask * cls_loss
                cls_loss = 0.5 * cls_loss

                reg_loss = torch.sum(reg_loss, dim=-1)
                reg_loss = reg_mask * reg_loss

                reg_loss = reg_loss.sum() / 32
                cls_loss = reg_loss.sum() / 32

                loss = cls_loss + reg_loss
                loss.backward()

                train_loss += loss.item()

                # Clip gradients and update parameters  
                torch.nn.utils.clip_grad_norm_(fe.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(rpn.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(roi.parameters(), 1.0)

                optimizer.step() 

                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
        fe.eval()
        rpn.eval()
        roi.eval()
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

                anc_box_list = anc_box_list.to(f'cuda:{rpn.device_ids[0]}')

                top_n = 1000

                start = time.time()
                corners, mask = rpn_to_roi(cls_scores=pred_cls_scores, pred_locs=pred_anch_locs, 
                                           anc_boxes=anc_box_list, nms_thresh=0.1, top_n=top_n)
                end = time.time()
                #print(f'IOU NMS time: {end - start}.')

                start = time.time()
                proposals = crp(fm, corners)
                end = time.time()
                #print(f'Crop proposals time: {end - start}.')

                pred_cls_scores, pred_anch_locs = roi(proposals)

                top_n_y = []
                top_n_bb_y = []
                final_mask = []
                y = y.squeeze()
                pred_cls_scores = pred_cls_scores.squeeze()
                for i in range(len(y)): 
                    c_i = pred_cls_scores[i]
                    y_i = y[i]
                    b_i = bb_y[i] 
                    m_i = mask[i]
                    _, sorted_idxs = torch.sort(c_i, descending=True)

                    top_n_y.append(y_i[sorted_idxs])
                    top_n_bb_y.append(b_i[sorted_idxs])
                    final_mask.append(m_i[sorted_idxs])

                top_n_y = torch.stack(top_n_y)
                top_n_bb_y = torch.stack(top_n_bb_y)
                final_mask = torch.stack(final_mask)

                if (top_n_y == 1.).sum() == 0: 
                    pos_weight = (top_n_y == 0).sum()
                else: 
                    pos_weight = (top_n_y == 0.).sum() / (top_n_y == 1.).sum()

                cls_loss = F.binary_cross_entropy_with_logits(pred_cls_scores, top_n_y, pos_weight=pos_weight, reduction='none')
                reg_loss = F.smooth_l1_loss(pred_anch_locs, top_n_bb_y, reduction='none', beta=1)

                final_mask &= (top_n_y != -1)

                reg_mask = final_mask.detach().clone()
                reg_mask &= (top_n_y == 1)

                reg_mask = reg_mask.float()
                final_mask = final_mask.float()

                cls_loss = final_mask * cls_loss.detach()
                cls_loss = 0.5 * cls_loss.detach()

                reg_loss = torch.sum(reg_loss, dim=-1)
                reg_loss = reg_mask * reg_loss.detach()

                reg_loss = reg_loss.mean()
                reg_loss.requires_grad = True

                cls_loss = cls_loss.mean()
                cls_loss.requires_grad = True

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