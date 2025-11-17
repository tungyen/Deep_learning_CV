import torch
import torch.nn as nn
import torch.nn.functional as F

from Object_detection_2d.SSD.model.vgg import VGG
from Object_detection_2d.SSD.model.post_processor import PostProcessor
from Object_detection_2d.SSD.model.anchors import PriorBox
from Object_detection_2d.SSD.utils.box_utils import *

class DetectionHead(nn.Module):
    def __init__(self, class_num, boxes_per_feature_map=[], input_channels=[]):
        super().__init__()
        self.class_num = class_num
        n_boxes = boxes_per_feature_map

        self.box_heads = nn.ModuleList()
        self.cls_heads = nn.ModuleList()
        for i, (n_box, input_channel) in enumerate(zip(n_boxes, input_channels)):
            self.box_heads.append(self.box_block(i, input_channel, n_box))
            self.cls_heads.append(self.cls_block(i, input_channel, n_box))

    def cls_block(self, level, input_channels, n_box):
        return nn.Conv2d(input_channels, n_box * self.class_num, kernel_size=3, stride=1, padding=1)
    
    def box_block(self, level, input_channels, n_box):
        return nn.Conv2d(input_channels, n_box * 4, kernel_size=3, stride=1, padding=1)
   
    def forward(self, feats):
        batch_size = feats[0].shape[0]
        boxes = []
        cls_logits = []
        for feat, box_head, cls_head in zip(feats, self.box_heads, self.cls_heads):
            boxes.append(box_head(feat).permute(0, 2, 3, 1).contiguous())
            cls_logits.append(cls_head(feat).permute(0, 2, 3, 1).contiguous())

        boxes = torch.cat([b.view(b.shape[0], -1) for b in boxes], dim=1).view(batch_size, -1, 4)
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.class_num)
        return boxes, cls_logits

class SSD(nn.Module):
    def __init__(self, backbone, head, prior, post_processor,
                 center_variance=0.1, size_variance=0.2):
        super().__init__()
        self.backbone = VGG(**backbone)
        self.det_head = DetectionHead(**head)
        self.prior_xy = PriorBox(**prior)()
        self.post_processor = PostProcessor(**post_processor)
        self.center_variance = center_variance
        self.size_variance = size_variance

    def forward(self, x, is_train=True):
        backbone_feats = self.backbone(x)
        boxes, cls_scores = self.det_head(backbone_feats)
        if is_train:
            return boxes, cls_scores
        self.prior_xy = self.prior_xy.to(x.device)
        cls_scores = F.softmax(cls_scores, dim=2)
        boxes = offset_to_cxcy(boxes, self.prior_xy, self.center_variance, self.size_variance)
        boxes = cxcy_to_xy(boxes)
        detections = (boxes, cls_scores)
        detections = self.post_processor(detections)
        return detections