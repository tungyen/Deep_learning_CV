import sys
import warnings

import torch
import torchvision
from packaging import version

if version.parse(torchvision.__version__) >= version.parse("0.3.0"):
    _nms = torchvision.ops.nms
else:
    warnings.warn('No NMS is available. Please upgrade torchvision to 0.3.0+')
    sys.exit(-1)

def nms(boxes, scores, nms_thres):
    keep = _nms(boxes, scores, nms_thres)
    return keep

def batched_nms(boxes, scores, idxs, iou_thres):
    if boxes.numel() == 0:
        return torch.empty((0, ), dtype=torch.int64, device=boxes.device)

    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_thres)
    return keep