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
    fe = torch.nn.DataParallel(fe, device_ids=[1,2,3])
    fe.to(f'cuda:{fe.device_ids[0]}')

    rpn = RPN(128, 512, 3)
    rpn = torch.nn.DataParallel(rpn, device_ids=[1,2,3])
    rpn.to(f'cuda:{rpn.device_ids[0]}')

    roi = ROI()
    roi.apply(weight_init)
    roi = torch.nn.DataParallel(roi, device_ids=[1,2,3])
    roi.to(f'cuda:{roi.device_ids[0]}')

    crp = CropProposals()
    crp = torch.nn.DataParallel(crp, device_ids=[1,2,3])
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
                fname, x, y, bb_y = data

                # This is used for logging 
                fname = list(fname)

                # Make labels correct size 
                y = y.unsqueeze(1)

                x = x.to(f'cuda:{fe.device_ids[0]}')
                y = y.to(f'cuda:{rpn.device_ids[0]}')
                bb_y = bb_y.to(f'cuda:{rpn.device_ids[0]}')
                            
                rpn_optimizer.zero_grad()
                roi_optimizer.zero_grad()
                # FE + RPN OPERATIONS 

                fm = fe(x)
                pred_anch_locs, pred_cls_scores = rpn(fm)
 
                sampled_pred, sampled_target = sample_anchor_boxes(pred_cls_scores, y)

                pos_weight = get_pos_weight_val(sampled_target)

                rpn_cls_loss = F.binary_cross_entropy_with_logits(sampled_pred, sampled_target, pos_weight=pos_weight, reduction='mean')
                rpn_cls_loss = 0.5 * rpn_cls_loss

                rpn_reg_loss = F.smooth_l1_loss(pred_anch_locs, bb_y, beta=1, reduction='none')

                mask = torch.where(y > 0, 1, 0)
                mask = mask.permute(0, 2, 1)

                rpn_reg_loss = rpn_reg_loss * mask
                rpn_reg_loss = rpn_reg_loss.sum() / (rpn_reg_loss != 0).sum()

                rpn_loss = rpn_cls_loss + rpn_reg_loss

                rpn_loss.backward(retain_graph=True)

                rpn_train_loss += rpn_loss.item()

                # BEGIN ROI OPERATIONS 

                top_n = 2500

                start = time.time()
                corners, mask, indexs = rpn_to_roi(cls_scores=pred_cls_scores, pred_locs=pred_anch_locs, 
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
                    y_i = y[i]
                    b_i = bb_y[i] 
                    m_i = mask[i]

                    top_n_y.append(y_i[indexs[i]])
                    top_n_bb_y.append(b_i[indexs[i]])
                    final_mask.append(m_i[indexs[i]])

                top_n_y = torch.stack(top_n_y)
                top_n_bb_y = torch.stack(top_n_bb_y)
                final_mask = torch.stack(final_mask)

                update_cm(top_n_y, pred_cls_scores, train_cm)

                final_mask = final_mask & (top_n_y != -1)
    
                pos_weight = get_pos_weight_val(top_n_y)

                roi_cls_loss = F.binary_cross_entropy_with_logits(pred_cls_scores, top_n_y, pos_weight=pos_weight, reduction='none')
                roi_cls_loss = roi_cls_loss.clone() * final_mask.float()
                roi_cls_loss = roi_cls_loss.sum() / (roi_cls_loss != 0).sum()
                roi_cls_loss = 0.5 * roi_cls_loss

                final_mask = final_mask & (top_n_y == 1)

                roi_reg_loss = F.smooth_l1_loss(pred_anch_locs, top_n_bb_y, reduction='none', beta=1)
                roi_reg_loss = roi_reg_loss.clone() * final_mask.unsqueeze(2).float()

                if (roi_reg_loss != 0).sum() == 0: 
                    roi_reg_loss = 0
                else: 
                    roi_reg_loss = roi_reg_loss.sum() / (roi_reg_loss != 0).sum()

                roi_loss = roi_reg_loss + roi_cls_loss

                roi_loss.backward()

                roi_train_loss += roi_loss.item()

                # Clip gradients and update parameters  
                torch.nn.utils.clip_grad_norm_(fe.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(rpn.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(roi.parameters(), 1.0)

                rpn_optimizer.step() 
                roi_optimizer.step()

                pbar.set_postfix(rpn_loss=rpn_loss.item(), roi_loss=roi_loss.item(), lr=rpn_optimizer.param_groups[0]['lr'])

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

            sampled_pred, sampled_target = sample_anchor_boxes(pred_cls_scores, y) 

            pos_weight = get_pos_weight_val(sampled_target)

            rpn_cls_loss = F.binary_cross_entropy_with_logits(sampled_pred, sampled_target, pos_weight=pos_weight, reduction='mean')
            rpn_cls_loss = 0.5 * rpn_cls_loss

            rpn_reg_loss = F.smooth_l1_loss(pred_anch_locs, bb_y, beta=1, reduction='none')

            mask = torch.where(y > 0, 1, 0)
            mask = mask.permute(0, 2, 1)

            rpn_reg_loss = rpn_reg_loss * mask
            rpn_reg_loss = rpn_reg_loss.sum() / (rpn_reg_loss != 0).sum() 

            rpn_loss = rpn_cls_loss + rpn_reg_loss

            rpn_val_loss += rpn_loss.item()

            # BEGIN ROI OPERATIONS 

            top_n = 250

            start = time.time()
            corners, mask, indexs = rpn_to_roi(cls_scores=pred_cls_scores, pred_locs=pred_anch_locs, 
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
                y_i = y[i]
                b_i = bb_y[i] 
                m_i = mask[i]

                top_n_y.append(y_i[indexs[i]])
                top_n_bb_y.append(b_i[indexs[i]])
                final_mask.append(m_i[indexs[i]])

            top_n_y = torch.stack(top_n_y)
            top_n_bb_y = torch.stack(top_n_bb_y)
            final_mask = torch.stack(final_mask)

            update_cm(top_n_y, pred_cls_scores, val_cm)

            final_mask &= (top_n_y != -1)
            
            pos_weight = get_pos_weight_val(top_n_y)

            roi_cls_loss = F.binary_cross_entropy_with_logits(pred_cls_scores, top_n_y, pos_weight=pos_weight, reduction='none')
            roi_cls_loss = roi_cls_loss.clone() * final_mask.float()
            roi_cls_loss = roi_cls_loss.sum() / (roi_cls_loss != 0).sum()
            roi_cls_loss = 0.5 * roi_cls_loss

            final_mask = final_mask & (top_n_y == 1)

            roi_reg_loss = F.smooth_l1_loss(pred_anch_locs, top_n_bb_y, reduction='none', beta=1)
            roi_reg_loss = roi_reg_loss.clone() * final_mask.unsqueeze(2).float()

            if (roi_reg_loss != 0).sum() == 0: 
                roi_reg_loss = 0
            else: 
                roi_reg_loss = roi_reg_loss.sum() / (roi_reg_loss != 0).sum()

            roi_loss = roi_reg_loss + roi_cls_loss

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