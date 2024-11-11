import torch
import numpy as np 
from src.util.util import makeDataLoaders
from src.model.feature_extractor import FeatureExtractor

def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 6] where each row is z1, y1, x1, z2, y2, x2
    deltas: [N, 6] where each row is [dz, dy, dx, log(dd), log(dh), log(dw)]
    """
    # Convert to z, y, x, d, h, w
    depth = boxes[:, 3] - boxes[:, 0]
    height = boxes[:, 4] - boxes[:, 1]
    width = boxes[:, 5] - boxes[:, 2]
    center_z = boxes[:, 0] + 0.5 * depth
    center_y = boxes[:, 1] + 0.5 * height
    center_x = boxes[:, 2] + 0.5 * width
    # Apply deltas
    center_z += deltas[:, 0] * depth
    center_y += deltas[:, 1] * height
    center_x += deltas[:, 2] * width
    depth *= torch.exp(deltas[:, 3])
    height *= torch.exp(deltas[:, 4])
    width *= torch.exp(deltas[:, 5])
    # Convert back to z1, y1, x1, z2, y2, x2
    z1 = center_z - 0.5 * depth
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z2 = z1 + depth
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([z1, y1, x1, z2, y2, x2], dim=1)
    return result


def clip_boxes(boxes, window):
    """boxes: [N, 6] each col is z1, y1, x1, z2, y2, x2
    window: [6] in the form z1, y1, x1, z2, y2, x2
    """
    boxes = torch.stack(
        [boxes[:, 0].clamp(float(window[0]), float(window[3])),
         boxes[:, 1].clamp(float(window[1]), float(window[4])),
         boxes[:, 2].clamp(float(window[2]), float(window[5])),
         boxes[:, 3].clamp(float(window[0]), float(window[3])),
         boxes[:, 4].clamp(float(window[1]), float(window[4])),
         boxes[:, 5].clamp(float(window[2]), float(window[5]))], 1)
    return boxes


def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.
    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dz, dy, dx, log(dd), log(dh), log(dw))]
    Returns:
        Proposals in normalized coordinates [batch, rois, (z1, y1, x1, z2, y2, x2)]
    """

    # Currently only supports batchsize 1
    inputs[0] = inputs[0].squeeze(0)
    inputs[1] = inputs[1].squeeze(0)

    # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
    scores = inputs[0][:, 1]

    # Box deltas [batch, num_rois, 6]
    deltas = inputs[1]
    std_dev = torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 6])).float()
    if config.GPU_COUNT:
        std_dev = std_dev.cuda()
    deltas = deltas * std_dev

    # Improve performance by trimming to top anchors by score
    # and doing the rest on the smaller subset.
    pre_nms_limit = min(config.PRE_NMS_LIMIT, anchors.size()[0])
    scores, order = scores.sort(descending=True)
    order = order[:pre_nms_limit]
    scores = scores[:pre_nms_limit]
    deltas = deltas[order.detach(), :]
    anchors = anchors[order.detach(), :]

    # Apply deltas to anchors to get refined anchors.
    # [batch, N, (z1, y1, x1, z2, y2, x2)]
    boxes = apply_box_deltas(anchors, deltas)

    # Clip to image boundaries. [batch, N, (z1, y1, x1, z2, y2, x2)]
    height, width, depth = config.IMAGE_SHAPE[:3]
    window = np.array([0, 0, 0, depth, height, width]).astype(np.float32)
    boxes = clip_boxes(boxes, window)

    # Non-max suppression
    keep = non_max_suppression(boxes.cpu().detach().numpy(),
                                     scores.cpu().detach().numpy(), nms_threshold, proposal_count)
    keep = torch.from_numpy(keep).long()
    boxes = boxes[keep, :]

    # Normalize dimensions to range of 0 to 1.
    norm = torch.from_numpy(np.array([depth, height, width, depth, height, width])).float()
    if config.GPU_COUNT:
        norm = norm.cuda()
    normalized_boxes = boxes / norm

    # Add back batch dimension
    normalized_boxes = normalized_boxes.unsqueeze(0)

    return normalized_boxes

############################################################
#  Bounding Boxes
############################################################

def compute_iou(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [z1, y1, x1, z2, y2, x2]
    boxes: [boxes_count, (z1, y1, x1, z2, y2, x2)]
    box_volume: float. the volume of 'box'
    boxes_volume: array of depth boxes_count.

    Note: the volumes are passed in rather than calculated here for
          efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection volumes
    z1 = np.maximum(box[0], boxes[:, 0])
    z2 = np.minimum(box[3], boxes[:, 3])
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[4], boxes[:, 4])
    x1 = np.maximum(box[2], boxes[:, 2])
    x2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / (union + 1e-6)
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # volumes of anchors and GT boxes
    volume1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    volume2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, volume2[i], volume1)
    return overlaps


def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (z1, y1, x1, z2, y2, x2)]
    """

    depth = box[:, 3] - box[:, 0]
    height = box[:, 4] - box[:, 1]
    width = box[:, 5] - box[:, 2]
    center_z = box[:, 0] + 0.5 * depth
    center_y = box[:, 1] + 0.5 * height
    center_x = box[:, 2] + 0.5 * width

    gt_depth = gt_box[:, 3] - gt_box[:, 0]
    gt_height = gt_box[:, 4] - gt_box[:, 1]
    gt_width = gt_box[:, 5] - gt_box[:, 2]
    gt_center_z = gt_box[:, 0] + 0.5 * gt_depth
    gt_center_y = gt_box[:, 1] + 0.5 * gt_height
    gt_center_x = gt_box[:, 2] + 0.5 * gt_width

    dz = (gt_center_z - center_z) / depth
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dd = torch.log(gt_depth / depth)
    dh = torch.log(gt_height / height)
    dw = torch.log(gt_width / width)

    result = torch.stack([dz, dy, dx, dd, dh, dw], dim=1)
    return result


def non_max_suppression(boxes, scores, threshold, max_num):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (z1, y1, x1, z2, y2, x2)]. Notice that (z2, y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    max_num: Int. The max number of boxes to keep.
    Return the index of boxes to keep.
    """
    # Compute box volumes
    z1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x1 = boxes[:, 2]
    z2 = boxes[:, 3]
    y2 = boxes[:, 4]
    x2 = boxes[:, 5]
    volume = (z2 - z1) * (y2 - y1) * (x2 - x1)

    # Get indices of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        if len(pick) >= max_num:
            break
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], volume[i], volume[ixs[1:]])
        # Identify boxes with IoU over the threshold. This returns indices into ixs[1:],
        # so add 1 to get indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def denorm_boxes_graph(boxes, size):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (z1, y1, x1, z2, y2, x2)] in normalized coordinates
    size: [depth, height, width], the size to denorm to.

    Note: In pixel coordinates (z2, y2, x2) is outside the box.
          But in normalized coordinates it's inside the box.

    Returns:
        [..., (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    """
    d, h, w = size
    scale = torch.Tensor([d, h, w, d, h, w]).cuda()
    denorm_boxes = torch.mul(boxes, scale)
    return denorm_boxes

def batch_slice(inputs, graph_fn, batch_size, names=None):
    """
    Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension size
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [torch.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result

if __name__ == "__main__": 
    train_loader, val_loader = makeDataLoaders(batch_size=32)

    fe = FeatureExtractor()
    fe.to('cuda:0')

    for data in train_loader: 
        fname, x, y, bb_y = data

        x = x.to('cuda:0')
        fm = fe(x)

        print(fm.shape)

        exit()