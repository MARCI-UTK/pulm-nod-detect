import os
import torch 
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import time 

from src.model.rpn_loss import RegLoss, ClsLoss, ValClsLoss, NatureClsLoss
from src.util.util import *

from rpn import RPN, get_centers, get_anc_boxes
from roi import ROI, CropProposals
from src.model.feature_extractor import FeatureExtractor

import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
from clearml import Task, Logger

task = Task.init(project_name="Pulmonary Nodule Detection", task_name="ROI w/ Augmented Data + Regularization (0.0001) + Momentum (Init. LR = 0.01)")
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

    fe = FeatureExtractor()
    fe.apply(weight_init)
    fe = torch.nn.DataParallel(fe, device_ids=[0,1,2,3])
    fe.to(f'cuda:{fe.device_ids[0]}')

    rpn = RPN(128, 512, 3)
    rpn = torch.nn.DataParallel(rpn, device_ids=[0,1,2,3])
    rpn.to(f'cuda:{rpn.device_ids[0]}')

    roi = ROI()
    roi = torch.nn.DataParallel(roi, device_ids=[0,1,2,3])
    roi.to(f'cuda:{roi.device_ids[0]}')

    cropper = CropProposals()
    cropper = torch.nn.DataParallel(cropper, device_ids=[0,1,2,3])
    cropper.to(f'cuda:{cropper.device_ids[0]}')

    # Create optimizer and LR scheduler 
    optimizer = optim.SGD(list(fe.parameters()) + list(rpn.parameters()), lr=0.01, momentum=0.9, weight_decay=0.0001)

    # Define loss functions 
    reg_loss_func = RegLoss()
    cls_loss_func = ClsLoss()

    train_losses = []  

    anc_box_list = anc_box_list.to(f'cuda:{roi.device_ids[0]}') 
    
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

                print('Passing crop to FE...')
                start = time.time()
                fm = fe(x)
                end = time.time()
                print(f'Feature extraction took {end - start}s.')

                print("Passing feature map to RPN...")
                start = time.time()
                pred_anch_locs, pred_cls_scores = rpn(fm)
                end = time.time()
                print(f'RPN took {end - start}s.')

                """
                print("Creating corner points from proposals...")
                start = time.time()
                corners = make_corners(anc_box_list, pred_anch_locs)
                end = time.time()
                print(f'Making corners took {end - start}s.')
                corners = corners.to(f'cuda:{roi.device_ids[0]}')
                """

                print("Thresholding proposals...")
                start = time.time()
                pred_cls_scores, y, pred_anch_locs, bb_y, corners = threshold_proposals(pred_anch_locs, 
                                                                                        bb_y, 
                                                                                        pred_cls_scores, 
                                                                                        y, 
                                                                                        corners)

                end = time.time()

                print(f'Thresholding proposals took {end - start}s.')

                print("Performing non-maximum suppression...")
                start = time.time()
                pred_cls_scores, y, corners = nms(pred_cls_scores, y, corners)
                end = time.time()
                print(f'NMS took {end - start}s.')

                print("Cropping proposals...")
                start = time.time()
                proposals = cropper(fm, corners, f'cuda:{roi.device_ids[0]}')
                end = time.time()
                print(f'Cropping proposals took {end - start}s.')
                proposals = proposals.view(-1, 128, 2, 2, 2).contiguous()

                print("Passing proposals to ROI module...")
                start = time.time()
                pred_anch_locs, pred_cls_scores = roi(proposals)
                end = time.time()
                print(f'ROI module took {end - start}s.')
                #update_cm(y, pred_cls_scores, train_cm) 

                #sampled_pred, sampled_targets = sample_anchor_boxes(pred_cls_scores, y)

                print("Calculating losses...")

                if (y == 1.).sum() == 0: 
                    pos_weight = (y == 0).sum()
                else: 
                    pos_weight = (y == 0.).sum() / (y == 1.).sum()

                loss = F.binary_cross_entropy_with_logits(input=pred_cls_scores, target=y, pos_weight=pos_weight, reduction='none')
                
                mask = torch.where(y < 0, 0, 1)

                cls_loss = 0
                for i in range(len(loss)): 
                    cls_loss += 0.5 * (loss[i] * mask[i]).sum()

                cls_loss /= len(loss)

                loss = F.smooth_l1_loss(input=pred_anch_locs, target=bb_y, beta=1, reduction='none')
                
                #mask = mask.permute(0, 2, 1)
                reg_loss = 0
                for i in range(len(loss)): 
                    reg_loss += (loss[i] * mask[i]).sum()

                reg_loss /= len(loss)

                print(cls_loss.requires_grad, reg_loss.requires_grad)

                loss = cls_loss + reg_loss
                loss.backward()

                print(loss.item())
                exit()

                train_loss += loss.item()

                iteration = e * len(train_loader) + idx
                logger.report_scalar(
                    "Train Cls. vs. Reg. Loss", "Reg. Loss", iteration=iteration, value=reg_loss,
                )
                logger.report_scalar(
                    "Train Cls. vs. Reg. Loss", "Cls. Loss", iteration=iteration, value=cls_loss,
                )
                logger.report_scalar(
                    "Train Total Loss", "Loss", iteration=iteration, value=loss.item(),
                )

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

                corners = make_corners(anc_box_list, pred_anch_locs)
                corners = corners.to(f'cuda:{roi.device_ids[0]}')

                pred_cls_scores, y, pred_anch_locs, bb_y, corners = threshold_proposals(pred_anch_locs, 
                                                                                        bb_y, 
                                                                                        pred_cls_scores, 
                                                                                        y, 
                                                                                        corners)
                
                pred_cls_scores, y, corners = nms(pred_cls_scores, y, corners)

                proposals = cropper(fm, corners, f'cuda:{roi.device_ids[0]}')

                print(proposals.shape)

                pred_anch_locs, pred_cls_scores = roi(proposals)
                update_cm(y, pred_cls_scores, train_cm) 
            
                if (y == 1.).sum() == 0: 
                    pos_weight = (y == 0).sum()
                else: 
                    pos_weight = (y == 0.).sum() / (y == 1.).sum()

                loss = F.binary_cross_entropy_with_logits(input=pred_anch_locs, target=y, pos_weight=pos_weight, reduction='none')
                
                mask = torch.where(y < 0, 0, 1)

                cls_loss = 0
                for i in range(len(loss)): 
                    cls_loss += 0.5 * (loss[i] * mask[i]).sum()

                cls_loss /= len(loss)

                loss = F.smooth_l1_loss(input=pred_anch_locs, target=bb_y, beta=1, reduction='none')
                
                #mask = mask.permute(0, 2, 1)
                reg_loss = 0
                for i in range(len(loss)): 
                    reg_loss += (loss[i] * mask[i]).sum()

                reg_loss /= len(loss) 

                val_loss += cls_loss.item() + reg_loss.item()

                iteration = e * len(val_loader) + idx
                logger.report_scalar(
                    "Val. Cls. vs. Reg. Loss", "Reg. Loss", iteration=iteration, value=reg_loss,
                )
                logger.report_scalar(
                    "Val. Cls. vs. Reg. Loss", "Cls. Loss", iteration=iteration, value=cls_loss,
                )
                logger.report_scalar(
                    "Val. Total Loss", "Loss", iteration=iteration, value=loss.item(),
                )
                
        if e > 50: 
            optimizer.param_groups[0]['lr'] = 0.001

        # Take average of losses using the number of batches
        epoch_train_loss = train_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)

        # Print epoch losses and confusion matrix statistics
        print(f'finished epoch {e}. Train Loss: {epoch_train_loss}. Val Loss: {epoch_val_loss}')
        print(f'train [tp, tn, fp, fn]: [{train_cm[0]}, {train_cm[1]}, {train_cm[2]}, {train_cm[3]}].')
        print(f'val   [tp, tn, fp, fn]: [{val_cm[0]}, {val_cm[1]}, {val_cm[2]}, {val_cm[3]}].')

    return 

if __name__ == "__main__": 
    main()