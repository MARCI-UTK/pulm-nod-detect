import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from matplotlib.patches import Rectangle

def world_2_voxel(world_point: tuple, world_origin: tuple, spacing: tuple) -> tuple: 
    world_x, world_y, world_z, diameter = world_point
    origin_x, origin_y, origin_z        = world_origin
    spacing_x, spacing_y, spacing_z     = spacing

    voxel_x = (world_x - origin_x) // spacing_x
    voxel_y = (world_y - origin_y) // spacing_y
    voxel_z = (world_z - origin_z) // spacing_z

    voxel_diameter = diameter // spacing_x

    voxel_point = (int(voxel_x), int(voxel_y), int(voxel_z), int(voxel_diameter))

    return(voxel_point)

# Adjust bounding box to match resized scan shape
def adjust_bbox(original_coords, original_shape, new_size):
    orig_x, orig_y, orig_z = original_shape

    x_scale = new_size / orig_x
    y_scale = new_size / orig_y
    z_scale = new_size / orig_z

    scales = [x_scale, y_scale, z_scale]

    new_coords = [original_coords[i] * scales[i] for i in range(3)]
    new_coords = [round(x) for x in new_coords]

    new_coords.append(original_coords[3] * (new_size / orig_x)) 

    return new_coords

def xyzd_2_2corners(c): 
    x, y, z, d = c 
    r = d / 2

    c_1 = [x - r, y - r, z - r]
    c_2 = [x + r, y + r, z + r]

    return [c_1, c_2]

def visualize_attention_map(im, mask, out, loc, e, idx): 
    x, y, z, d = loc
    
    out = torch.sigmoid(out)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 24))

    res = (im * out).type(torch.uint8)

    ax1.set_title('Original')
    ax2.set_title('Mask')
    ax3.set_title('Attention Map Overlay')
    _ = ax1.imshow(im[int(z.item())], cmap=plt.bone())
    _ = ax2.imshow(mask[int(z.item())], cmap=plt.bone())
    _ = ax3.imshow(res[int(z.item())], cmap=plt.bone())

    x_rect = x - (d / 2)
    y_rect = y - (d / 2)

    rectangle_position = (x_rect, y_rect)
    rectangle_size = (d, d)     
    rectangle = Rectangle(rectangle_position, *rectangle_size, edgecolor='red', facecolor='none', linewidth=1)
    ax1.add_patch(rectangle)

    plt.savefig(f'imgs/attn_map_output_{e}_{idx}.png')
    plt.close()

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target, epsilon=1e-6):
        # Apply sigmoid to predictions to get probabilities
        pred = torch.sigmoid(pred)
        
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        numerator = 2 * (pred * target).sum(1)
        denominator = pred.sum(-1) + target.sum(-1)
        loss = 1 - (numerator + epsilon) / (denominator + epsilon) 
        
        return torch.mean(loss)
    
class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    """
    def forward(self, pred, target, alpha, gamma):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        
        return focal_loss.mean()