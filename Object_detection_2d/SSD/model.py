import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from math import sqrt

from Object_detection_2d.SSD.utils import decimate

class VggBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.load_pretrained_weights()

    def forward(self, image):
        feats = []
        out = F.relu(self.conv1_1(image))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)

        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out)

        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)

        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        feats.append(out)
        out = self.pool4(out)

        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)

        out = F.relu(self.conv6(out))
        conv7_feats = F.relu(self.conv7(out))
        feats.append(conv7_feats)
        return feats

    def load_pretrained_weights(self):
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())

        for i, param in enumerate(param_names[:-4]):
            state_dict[param] = pretrained_param_names[pretrained_param_names[i]]

        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])

        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])
        self.load_state_dict(state_dict)

class AuxiliaryConvolution(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv2d_init()

    def conv2d_init(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.0)
    
    def forward(self, conv7_feats):
        feats = []
        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out))
        feats.append(out)

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        feats.append(out)

        out = F.relu(self.conv10_1(out))
        out = F.relu(self.conv10_2(out))
        feats.append(out)

        out = F.relu(self.conv11_1(out))
        conv11_2_feats = F.relu(self.conv11_2(out))
        feats.append(conv11_2_feats)
        return feats

class DetectionHead(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        n_boxes = [4, 6, 6, 6, 4, 4]
        input_channels_list = [512, 1024, 512, 256, 256, 256]

        self.box_heads = []
        for n_box, input_channels in zip(n_boxes, input_channels_list):
            self.box_heads.append(nn.Conv2d(input_channels, 4 * n_box, kernel_size=3, padding=1))

        self.cls_heads = []
        for n_box, input_channels in zip(n_boxes, input_channels_list):
            self.box_heads.append(nn.Conv2d(input_channels, class_num * n_box, kernel_size=3, padding=1))

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
            box_res = box_res.view(batch_size, 4, -1).contiguous()
            boxes.append(box_res)

            cls_res = self.cls_heads[i](feat)
            cls_res = cls_res.view(batch_size, self.class_num, -1).contiguous()
            classes.append(cls_res)

        boxes = torch.cat(boxes, dim=1)
        classes = torch.cat(classes, dim=1)
        return boxes, classes

class SSD(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num

        self.backbone = VggBackbone()
        self.aux_convs = AuxiliaryConvolution()
        self.det_head = DetectionHead(class_num)

        self.rescale_factor = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        self.prior_boxes_center = self.create_prior_boxes()

    def forward(self, x):
        backbone_feats = self.backbone(x)
        norm = backbone_feats[0].pow(2).sum(dim=1, keepdim=True).sqrt()
        backbone_feats[0] = backbone_feats[0].clone() / norm * self.rescale_factor
        
        aux_feats = self.aux_convs(backbone_feats[1])
        boxes, cls_scores = self.det_head(backbone_feats + aux_feats)
        return boxes, cls_scores

    def create_prior_boxes(self):
        feats_map_dims = [38, 19, 10, 5, 3, 1]
        scales = [0.1, 0.2, 0.375, 0.55, 0.725, 0.9]
        aspect_ratios = [
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 3.0, 0.5, 0.333],
            [1.0, 2.0, 3.0, 0.5, 0.333],
            [1.0, 2.0, 3.0, 0.5, 0.333],
            [1.0, 2.0, 0.5],
            [1.0, 2.0, 0.5]
        ]
        feats_num = len(feats_map_dims)
        prior_boxes = []
        
        for i in range(feats_num):
            dim = feats_map_dims[i]
            scale = scales[i]
            aspect_ratio = aspect_ratios[i]
            for r in range(dim):
                for c in range(dim):
                    cx = (c + 0.5) / dim
                    cy = (r + 0.5) / dim
                    
                    for ratio in aspect_ratio:
                        prior_boxes.append([cx, cy, scale * sqrt(ratio), scale / sqrt(ratio)])
                        
                        if ratio == 1.0:
                            if i != feats_num - 1:
                                additional_scale = sqrt(scale * scales[i+1])
                            else:
                                additional_scale = 1.0
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])
                            
        prior_boxes = torch.FloatTensor(prior_boxes)
        return prior_boxes