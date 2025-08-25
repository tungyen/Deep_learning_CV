import torch
import torch.nn as nn
import torch.nn.functional as F

from Object_detection_2d.SSD.model.vgg import VGG
from Object_detection_2d.SSD.model.post_processor import PostProcessor
from Object_detection_2d.SSD.model.anchors import PriorBox
from Object_detection_2d.SSD.utils.box_utils import *

import time

class DetectionHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.class_num = args['class_num']
        n_boxes = args['prior']['boxes_per_feature_map']
        input_channels_list = args['backbone']['out_channels']

        self.box_heads = []
        for n_box, input_channels in zip(n_boxes, input_channels_list):
            self.box_heads.append(nn.Conv2d(input_channels, 4 * n_box, kernel_size=3, padding=1))
        self.box_heads = nn.ModuleList(self.box_heads)
        self.cls_heads = []
        for n_box, input_channels in zip(n_boxes, input_channels_list):
            self.cls_heads.append(nn.Conv2d(input_channels, self.class_num * n_box, kernel_size=3, padding=1))
        self.cls_heads = nn.ModuleList(self.cls_heads)
        self.init_conv2d()

    def init_conv2d(self):
        for modules in self.children():
            for c in modules:
                if isinstance(c, nn.Conv2d):
                    nn.init.xavier_uniform_(c.weight)
                    nn.init.constant_(c.bias, 0.0)
   
    def forward(self, feats):
        assert(len(feats) == 6)
        batch_size = feats[0].shape[0]
        boxes = []
        classes = []
        for i, feat in enumerate(feats):
            box_res = self.box_heads[i](feat)
            boxes.append(box_res.permute(0, 2, 3, 1).contiguous())
            cls_res = self.cls_heads[i](feat)
            classes.append(cls_res.permute(0, 2, 3, 1).contiguous())

        boxes = torch.cat([b.view(batch_size, -1) for b in boxes], dim=1).view(batch_size, -1, 4)
        classes = torch.cat([c.view(batch_size, -1) for c in classes], dim=1).view(batch_size, -1, self.class_num)
        return boxes, classes

class SSD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = VGG(args)
        self.det_head = DetectionHead(args)
        self.prior_xy = PriorBox(args)()
        self.post_processor = PostProcessor(args)
        self.center_variance = args['center_variance']
        self.size_variance = args['size_variance']

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