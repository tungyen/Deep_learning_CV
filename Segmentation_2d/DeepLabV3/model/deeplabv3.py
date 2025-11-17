import torch
from torch import nn
from torch.nn import functional as F

from Segmentation_2d.utils import set_bn_momentum, initialize_weights
from Segmentation_2d.DeepLabV3.model import resnet
from Segmentation_2d.DeepLabV3.model.utils import IntermediateLayerGetter

class DeepLabV3(nn.Module):
    def __init__(self, in_channel=2048, class_num=21, backbone="resnet101", mid_channel=256, out_stride=16, pretrained_backbone=True,
                 bn_momentum=None, weight_init=None):
        super(DeepLabV3, self).__init__()
        if out_stride==8:
            replace_stride_with_dilation=[False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation=[False, False, True]
            aspp_dilate = [6, 12, 18]
            
        return_layers = {'layer4': 'out'}
        backbone = resnet.__dict__[backbone](pretrained=pretrained_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
        
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.classifier = DeepLabHead(in_channel, mid_channel, class_num, aspp_dilate)

        if bn_momentum is not None:
            set_bn_momentum(self.backbone, momentum=bn_momentum)

        if weight_init is not None:
            self.classifier.apply(initialize_weights(weight_init))
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
    
    
class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channel, class_num, backbone, mid_channel=256, out_stride=16, pretrained_backbone=True,
                 bn_momentum=None, weight_init=None):
        super(DeepLabV3Plus, self).__init__()
        if out_stride==8:
            replace_stride_with_dilation=[False, True, True]
            aspp_dilate = [12, 24, 36]
        else:
            replace_stride_with_dilation=[False, False, True]
            aspp_dilate = [6, 12, 18]
            
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        backbone = resnet.__dict__[backbone](pretrained=pretrained_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
        
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.classifier = DeepLabHeadV3Plus(in_channel, mid_channel, class_num, aspp_dilate=aspp_dilate)
        
        if bn_momentum is not None:
            set_bn_momentum(self.backbone, momentum=bn_momentum)

        if weight_init is not None:
            self.classifier.apply(initialize_weights(weight_init))

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
    
    
class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channel, low_level_channel, class_num, mid_channel=256, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(low_level_channel, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.aspp = ASPP(in_channel, mid_channel, aspp_dilate)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, class_num, 1)
        )
        self.init_weight()
        
    def forward(self, feature):
        out = self.proj(feature['low_level'])
        out_feat = self.aspp(feature['out'])
        out_feat = F.interpolate(out_feat, size=out.shape[2:], mode='bilinear', align_corners=False)
        return self.classifier(torch.cat([out, out_feat], dim=1))
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
class DeepLabHead(nn.Module):
    def __init__(self, in_channel, out_channel, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channel, out_channel, aspp_dilate),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, num_classes, 1)
        )
        self._init_weight()

    def forward(self, x):
        return self.classifier(x['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, dilation, in_channel, bias),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias)
        )
        
        self._init_weight()
        
    def forward(self, x):
        return self.layer(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
class ASPPConv(nn.Sequential):
    def __init__(self, in_channel, out_channel, dilation):
        modules = [
            nn.Conv2d(in_channel, out_channel, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)
        

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channel, out_channel):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    
    
class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel, atrous_rates):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channel, out_channel, rate1))
        modules.append(ASPPConv(in_channel, out_channel, rate2))
        modules.append(ASPPConv(in_channel, out_channel, rate3))
        modules.append(ASPPPooling(in_channel, out_channel))

        self.convs = nn.ModuleList(modules)

        self.proj = nn.Sequential(
            nn.Conv2d(5 * out_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.proj(res)
    
    
def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                    module.out_channels, 
                                    module.kernel_size,
                                    module.stride,
                                    module.padding,
                                    module.dilation,
                                    module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module