from rpn import get_centers, get_anc_boxes
from src.util.util import get_iou, xyzd_2_2corners, corners_2_xyzd, scanPathToId
from src.util.iou_3d import box3d_iou, get_3d_box

import os
import numpy as np 
import matplotlib.pyplot as plt

dataPath = '/data/marci/dlewis37/luna16/'
paths = [os.path.join(dataPath, 'dataset', f) for f in os.listdir(os.path.join(dataPath, 'dataset'))]    

def assign_invalid(corners, labels):
    for idx, c in enumerate(corners): 
        c1 = c[0]
        c2 = c[1]

        value = 0
        for i in range(3): 
            if c1[i] < 0 or c2[i] < 0: 
                value = -1
                break

            if c1[i] > 96 or c2[i] > 96:
                value = -1
                break
        
        if value == -1: 
            labels[idx] = value

    return labels

def assign_pos_neg(corners, labels, gt_box): 
    xyzd_corners = [corners_2_xyzd(i) for i in corners]

    for idx, c in enumerate(xyzd_corners): 
        if labels[idx] == -1: 
            continue
                    
        iou = get_iou(c, gt_box)

        if iou > 0.5: 
            labels[idx] = 1
        elif iou < 0.02: 
            labels[idx] = 0
        else: 
            labels[idx] = -1

    return labels

def make_final_locations(corners, labels, gt_box): 
    idxs = np.where(labels == 1)[0]
    final_locs = np.zeros(shape=(len(corners), 4))

    if len(idxs) < 1: 
        return final_locs
    
    for i in idxs: 
        c = corners[i]
        
        x, y, z, d_gt = gt_box
        x_pred, y_pred, z_pred, d_anc = c

        dx = (x - x_pred) / d_anc
        dy = (y - y_pred) / d_anc
        dz = (z - z_pred) / d_anc 
        dw = np.log(d_gt / d_anc)

        tmp = [dx, dy, dz, dw]

        for j in range(4): 
            final_locs[i, j] = tmp[j]

    return final_locs

def assign_labels(paths):
    centers = get_centers(orig_width=96, feat_width=24)
    corners = get_anc_boxes(centers=centers)

    xyzd_corners = [corners_2_xyzd(c) for c in corners]

    for p in paths: 
        anchor_labels = np.zeros(len(corners))
        anchor_labels = assign_invalid(corners=corners, labels=anchor_labels)

        data = np.load(p)
        img = data['img']
        gt_box = data['bbox']

        anchor_labels = assign_pos_neg(corners=corners, labels=anchor_labels, gt_box=gt_box)        

        final_locs = make_final_locations(corners=xyzd_corners, labels=anchor_labels, gt_box=gt_box)

        scanId = scanPathToId(p)
        outpath = os.path.join(dataPath, 'rpn_labels', f'{scanId}.npz')
        np.savez_compressed(file=outpath,
                            labels=anchor_labels,
                            locs=final_locs)
        
        print(f'wrote to {outpath}') 

    return 

if __name__ == "__main__": 
    assign_labels(paths=paths)