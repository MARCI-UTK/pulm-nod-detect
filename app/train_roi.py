import os
import torch 
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from src.model.rpn_loss import RegLoss, ClsLoss, ValClsLoss
from src.util.util import *

from rpn import RPN, get_centers, get_anc_boxes
from roi import ROI, CropProposals

from torch.autograd import Variable

import torch.nn.functional as F

from clearml import Task, Logger

task = Task.init(project_name="Pulmonary Nodule Detection", task_name="End-to-end RPN + ROI")
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

    roi = ROI()
    roi.apply(weight_init)
    roi = torch.nn.DataParallel(roi, device_ids=[0,1,2,3])
    roi.to(f'cuda:{roi.device_ids[0]}')

    proposal_fm_generator = CropProposals()

    # Create optimizer and LR scheduler 
    optimizer = optim.SGD(list(rpn.parameters()) + list(roi.parameters()), lr=0.0001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    # Define loss functions 
    reg_loss_func = RegLoss()
    cls_loss_func = ClsLoss()
    val_cls_loss_func = ValClsLoss()

    train_losses = []   
    val_losses = []

    cls_loss_norm = 0
    reg_loss_norm = 0
    rpn_loss_norm = 0

    epochs = 50
    for e in range(epochs): 

        train_loss = 0
        rpn_cm = [0, 0, 0, 0]
        roi_cm = [0, 0, 0, 0]
        
        rpn.train()
        roi.train()
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
                anc_box_list = anc_box_list.to(f'cuda:{rpn.device_ids[0]}')
                            
                optimizer.zero_grad()

                pred_anch_locs, pred_cls_scores, fm = rpn(x)
                update_cm(y, pred_cls_scores, rpn_cm) 
                              
                roi_y, roi_proposals, roi_gt_deltas, _ = generate_roi_input(pred_anch_locs, 
                                                                            bb_y, 
                                                                            pred_cls_scores, 
                                                                            y, 
                                                                            anc_box_list) 
                
                proposal_fms = proposal_fm_generator(fm, roi_proposals)
                proposal_fms = proposal_fms.to(f'cuda:{roi.device_ids[0]}')
                proposal_fms = proposal_fms.view(-1, 128, 2, 2, 2)

                roi_y = roi_y.to(f'cuda:{roi.device_ids[0]}')
                roi_y = roi_y.view(-1, 1)

                roi_gt_deltas = roi_gt_deltas.to(f'cuda:{roi.device_ids[0]}')
                roi_gt_deltas = roi_gt_deltas.view(-1, 4)

                cls, reg = roi(proposal_fms)

                if (roi_y == 1).sum() != 0: 
                    pos_weight = (roi_y == 0.).sum() / (roi_y == 1.).sum()
                else: 
                    pos_weight = (roi_y == 0).sum()

                cls_loss = F.binary_cross_entropy_with_logits(input=cls, target=roi_y, pos_weight=pos_weight)
                reg_loss = F.smooth_l1_loss(input=reg, target=roi_gt_deltas, beta=1)
                
                loss = (cls_loss) + (reg_loss)
                loss.to(f'cuda:{roi.device_ids[0]}')
                loss.backward()

                # Clip gradients and update parameters  
                torch.nn.utils.clip_grad_norm_(rpn.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(roi.parameters(), 1.0) 

                optimizer.step()

                train_loss += loss.item()

                # Update confustion matrix for current epoch 
                update_cm(roi_y, cls, roi_cm)

                pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

                logger.report_scalar(
                    'Training Loss', 'Loss', value=loss.item(), iteration=e * len(train_loader) + idx
                )
        
        if e > 23: 
            scheduler.step()

        # Take average of losses using the number of batches
        epoch_train_loss = train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # Print epoch losses and confusion matrix statistics
        print(f'finished epoch {e}. Train Loss: {epoch_train_loss}.')
        print(f'rpn [tp, tn, fp, fn]: [{rpn_cm[0]}, {rpn_cm[1]}, {rpn_cm[2]}, {rpn_cm[3]}]. rpn [tp, tn, fp, fn]: [{roi_cm[0]}, {roi_cm[1]}, {roi_cm[2]}, {roi_cm[3]}].')

    print(f'train losses: {train_losses}.')

    return 

if __name__ == "__main__": 
    main()