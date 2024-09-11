import os
import torch 
import numpy as np
import functools
import itertools
from torch.utils.data import DataLoader
from src.model.data import CropDataset
from itertools import chain, combinations

def scanPathToId(path: str) -> str: 
    return path.split('/')[-1][0:-4]

def windowImage(img: list, window: int, level: int) -> list: 
    min_hu = level - (window // 2)
    max_hu = level + (window // 2)

    windowed_img = np.copy(img)
    windowed_img = np.clip(windowed_img, min_hu, max_hu)

    return windowed_img

def worldToVoxel(world_point: tuple, world_origin: tuple, spacing: tuple) -> tuple: 
    world_x, world_y, world_z, diameter = world_point
    origin_x, origin_y, origin_z        = world_origin
    spacing_x, spacing_y, spacing_z     = spacing

    voxel_x = (world_x - origin_x) // spacing_x
    voxel_y = (world_y - origin_y) // spacing_y
    voxel_z = (world_z - origin_z) // spacing_z

    voxel_diameter = diameter // spacing_x

    voxel_point = (int(voxel_x), int(voxel_y), int(voxel_z), int(voxel_diameter))

    return(voxel_point)

def voxel_to_world(voxel_point: tuple, world_origin: tuple, spacing: tuple) -> tuple: 
    p       = list(voxel_point)
    origin  = list(world_origin)
    spacing = list(spacing)

    world_point = [p[i] * spacing[i] + origin[i] for i in range(3)]
    return tuple(world_point)

def get_iou(c1, c2): 
    # cube = [x, y, z, d]

    # How much to add/sub from center point to get corner
    offset1 = c1[3] / 2
    offset2 = c2[3] / 2

    # Get lower left (ll) and upper right (ur) coordinates for each box
    ll1 = [i - offset1 for i in c1[:3]]
    ur1 = [i + offset1 for i in c1[:3]]

    ll2 = [i - offset2 for i in c2[:3]]
    ur2 = [i + offset2 for i in c2[:3]]

    overlap = [max(0, min(ur1[i], ur2[i]) - max(ll1[i], ll2[i])) for i in range(3)]
    intersection = functools.reduce(lambda x, y : x * y, overlap)
    union = np.power(c1[3], 3) + np.power(c2[3], 3) - intersection

    iou = intersection / union
    
    return iou

def xyzd_2_2corners(c): 
    x, y, z, d = c 
    r = d / 2

    center = [x, y, z]

    c_1 = [x - r, y - r, z - r]
    c_2 = [x + r, y + r, z + r]

    return [c_1, c_2]

def corners_2_xyzd(c): 
    c1 = c[0]
    c2 = c[1]

    d = abs(c2[0] - c1[0])
    xyz = [c2[i] - (d // 2) for i in range(3)]
    
    xyz.append(d)

    return xyz

# targets.shape = [32, 1, 41472]
def sample_anchor_boxes(pred, targets): 

    # targets_batch.shape = [32, 41472]
    targets_batch = targets[:, 0, :]
    pred_batch = pred[:, 0, :]

    loss_targets = []
    loss_pred = []

    # Iterate through each sample in minibatch 
    for i in range(len(targets_batch)): 

        # Anchor box targets and predictions for sample (all anchor boxes)
        t = targets_batch[i]
        p = pred_batch[i]

        # Get indexes of positve and negative anchor boxes
        pos_idxs = (t == 1).nonzero(as_tuple=False)
        neg_idxs = (t == 0).nonzero(as_tuple=False)

        # Number of negative samples needed is (32 - # of positive samples) 
        neg_idxs = neg_idxs[torch.randint(0, len(neg_idxs), size=(32 - len(pos_idxs),))]

        neg_idxs = neg_idxs.squeeze(1)
        pos_idxs = pos_idxs.squeeze(1)

        # Select chosen anchor boxes, concatenate them together 
        # Do this for both targets and predictions 

        pos_t = torch.index_select(t, 0, pos_idxs)    
        neg_t = torch.index_select(t, 0, neg_idxs)
        final_t = torch.cat((pos_t, neg_t), 0)

        loss_targets.append(final_t)

        pos_p = torch.index_select(p, 0, pos_idxs)
        neg_p = torch.index_select(p, 0, neg_idxs)
        final_p = torch.cat((pos_p, neg_p), 0)

        loss_pred.append(final_p)

    # Stack the mini-batch back together for targets and predictions
    loss_targets = torch.stack(loss_targets)
    loss_pred = torch.stack(loss_pred)

    pos_count = (loss_targets == 1).sum().item()
    neg_count = (loss_targets == 0).sum().item()

    return loss_pred, loss_targets

def apply_bb_deltas(anc_box, deltas): 
    """
    rv = [anc_box[0] + deltas[0], 
          anc_box[1] + deltas[1], 
          anc_box[2] + deltas[2], 
          anc_box[3] * np.exp(deltas[3])] 
    """
    
    rv = [anc_box[i] + (anc_box[3] * deltas[i]) for i in range(3)]
    rv.append(anc_box[3] * np.exp(deltas[3]))

    return rv

def deltas_2_corners(anc_box, deltas): 
    rv = [anc_box[i] + (anc_box[3] * deltas[i]) for i in range(3)]
    rv.append(anc_box[3] * torch.exp(deltas[3]))

    x, y, z, d = rv 
    r = d / 2

    c_1 = [x - r, y - r, z - r]
    c_2 = [x + r, y + r, z + r]

    return [c_1, c_2]

"""
args: 
    - list of anchor box locations 
    - the suggested deltas output by regression head of RPN 

returns: 
    - an array of [batch_size, num_anchor_boxes, 2, 3]
    - the last 2 dimensions come from having 2 corners with 3 coordinates each 
"""
def make_corners(anc_boxes, pred_deltas): 
    corners = torch.zeros(pred_deltas.shape[0], pred_deltas.shape[1], 2, 3).cuda()

    # Loop through all crops in mini-batch  
    for i in range(len(pred_deltas)):
        tmp = [deltas_2_corners(anc_boxes[x], pred_deltas[i][x]) for x in range(len(pred_deltas[i]))]
        corners[i] = torch.tensor(tmp)

    print(corners.shape)

    return corners

def nms(pred_y, y, boxes): 
    pred_y = pred_y.squeeze()
    y = y.squeeze()

    for i in range(len(pred_y)): 
        #print(f'before: {(pred_y[i] != -1).sum()}')
        tmp_pred_y = pred_y[i]
        tmp_b = boxes[i]
        tmp_y = y[i]

        best_score = torch.argmax(tmp_pred_y)
        best_b = tmp_b[best_score, :]

        for j in range(len(tmp_b)): 
            if j == best_score: 
                continue 
        
            iou = get_iou(best_b, tmp_b[j])

            if iou > 0.1: 
                tmp_y[j] = -1.
                tmp_pred_y[j] = -1
                tmp_b[j] = torch.tensor([[-1., -1., -1.], [-1., -1., -1.]])
        
        pred_y[i] = tmp_pred_y
        boxes[i]  = tmp_b
        y[i] = tmp_y

        #print(f'after: {(pred_y[i] != -1).sum()}') 
    
    return pred_y, y, boxes
    
def threshold_proposals(pred_deltas, gt_deltas, cls_scores, targets, corners):
    cls_scores = cls_scores.squeeze()
    targets    = targets.squeeze()

    mask = ((targets != -1) & (cls_scores > 0.12))
    
    idxs = torch.masked_fill(torch.cumsum(mask.int(), dim=1), ~mask, 0)
    
    cls_scores = torch.scatter(input=torch.full_like(cls_scores, -1), dim=1, index=idxs, src=cls_scores)
    targets    = torch.scatter(input=torch.full_like(targets, -1), dim=1, index=idxs, src=targets)

    box_idxs = idxs[:, :, None].expand(pred_deltas.size(0), -1, pred_deltas.size(-1))

    pred_deltas = torch.scatter(input=torch.full_like(pred_deltas, -1), dim=1, index=box_idxs, src=pred_deltas)
    gt_deltas   = torch.scatter(input=torch.full_like(gt_deltas, -1), dim=1, index=box_idxs, src=gt_deltas)

    corner_idxs = idxs[:, :, None, None].expand(corners.size(0), -1, corners.size(-2), corners.size(-1)) 
    corners     = torch.scatter(input=torch.full_like(corners, -1), dim=1, index=corner_idxs, src=corners) 

    cls_scores = cls_scores.unsqueeze(1)
    targets    = targets.unsqueeze(1)

    #print(cls_scores.shape, targets.shape, pred_deltas.shape, gt_deltas.shape, corners.shape)

    return cls_scores, targets, pred_deltas, gt_deltas, corners

def generate_roi_input(pred_deltas, gt_deltas, cls_scores, y, anc_box_list):  
    roi_cls_scores = []
    roi_y = []
    roi_gt_deltas = []
    roi_anc_boxs = []

    roi_corners = torch.zeros((len(y), 200, 2, 3))

    for i in range(len(y)): 
        tmp_y = y.squeeze()
        tmp_cls_scores = cls_scores.squeeze()

        valid_idxs = (tmp_y[i] != -1)

        tmp_cls_scores = tmp_cls_scores[i][valid_idxs] 
        tmp_y = tmp_y[i][valid_idxs]
        tmp_anc_boxs = anc_box_list[valid_idxs]
        tmp_pred_deltas = pred_deltas[i][valid_idxs]
        tmp_gt_deltas = gt_deltas[i][valid_idxs]

        _, sorted_idxs = torch.sort(tmp_cls_scores, descending=True)

        tmp_anc_boxs = tmp_anc_boxs[sorted_idxs]
        tmp_pred_deltas = tmp_gt_deltas[sorted_idxs]

        for p in range(len(tmp_pred_deltas)): 
            modified_anc_box = apply_bb_deltas(tmp_anc_boxs[p].tolist(), tmp_pred_deltas[p].tolist())
            corners = xyzd_2_2corners(modified_anc_box)

            roi_corners[i][p] = torch.tensor(corners).cuda()
    
        roi_anc_boxs.append(anc_box_list[sorted_idxs])
        roi_gt_deltas.append(tmp_gt_deltas[sorted_idxs])

        roi_cls_scores.append(tmp_cls_scores[sorted_idxs])
        roi_y.append(tmp_y[sorted_idxs])
    
    roi_gt_deltas = torch.stack(roi_gt_deltas)
    roi_cls_scores = torch.stack(roi_cls_scores)
    roi_y = torch.stack(roi_y)

    return roi_y, roi_corners, roi_gt_deltas

def weight_init(m): 
    if isinstance(m, torch.nn.Conv3d): 
        torch.nn.init.xavier_uniform_(m.weight) 
    elif isinstance(m, torch.nn.Linear): 
        torch.nn.init.xavier_uniform_(m.weight)

def update_cm(y, pred, cm): 
    pred_binary = torch.where(pred > 0.5, 1., 0.)

    cm[0] += ((pred_binary == 1.) & (y == 1.)).sum().item()
    cm[1] += ((pred_binary == 0.) & (y == 0.)).sum().item()
    cm[2] += ((pred_binary == 1.) & (y == 0.)).sum().item()
    cm[3] += ((pred_binary == 0.) & (y == 1.)).sum().item()

def makeDataLoaders(): 
    dataPath = '/data/marci/dlewis37/luna16/'

    img_paths = [os.path.join(dataPath, 'dataset', f) for f in os.listdir(os.path.join(dataPath, 'dataset'))]
    label_paths = [os.path.join(dataPath, 'rpn_labels', f) for f in os.listdir(os.path.join(dataPath, 'rpn_labels'))]

    train_img_idxs = int(len(img_paths) * 0.8)
    train_img_paths = img_paths[:train_img_idxs - 1]
    val_img_paths   = img_paths[train_img_idxs:]

    train_label_idxs = int(len(img_paths) * 0.8)
    train_label_paths = label_paths[:train_label_idxs - 1]
    val_label_paths   = label_paths[train_label_idxs:]

    train_data = CropDataset(img_paths=train_img_paths, label_paths=train_label_paths)
    val_data   = CropDataset(img_paths=val_img_paths, label_paths=val_label_paths)
    batch_size = 32
    
    # 708 positive samples in training set 
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size, 
        shuffle=True
    )

    # 159 positive samples in validation set 
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size, 
        shuffle=True
    )

    return train_loader, val_loader

def powerset(iterable):
    "list(powerset([1,2,3])) --> [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))