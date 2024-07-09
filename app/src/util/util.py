import torch 
import numpy as np
import functools
import itertools
from numpy.random import default_rng

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

def xyzd_2_corners(c): 
    x, y, z, d = c 
    r = d // 2

    center = [x, y, z]
    corner_1 = [i - r for i in center]
    corner_2 = [i + r for i in center]

    return [corner_1, corner_2]

def corners_2_xyzd(c): 
    c1 = c[0]
    c2 = c[1]

    d = abs(c2[0] - c1[0])
    xyz = [c2[i] - (d // 2) for i in range(3)]
    
    xyz.append(d)

    return xyz

# targets.shape = [32, 1, 41472]
def sample_anchor_boxes(pred, targets, device): 

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