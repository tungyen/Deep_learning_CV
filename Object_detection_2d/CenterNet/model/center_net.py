import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
resnet_spec = {50: [3, 4, 6, 3],101: [3, 4, 23, 3],152: [3, 8, 36, 3]}

class Bottleneck(nn.Module):
    expansion=4

    def __init__(self, input_channels, output_channels, momentum=0.1, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channels, momentum=momentum)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels, momentum=momentum)
        self.conv3 = nn.Conv2d(output_channels, output_channels * self.expansion, kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(output_channels * self.expansion, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CenterNet(nn.Module):
    def __init__(self, args, head_channels=64, momentum=0.1):

        self.inter_channels = 64
        self.deconv_with_bias = False
        self.heads = {'hm': args['class_num'], 'wh': 2, 'reg': 2}
        self.num_layers = args['num_layers']
        self.momentum = momentum
        layers = resnet_spec[args['num_layers']]

        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.deconv_layers = self._make_deconv_layer()

        self.hm = nn.Sequential(
            nn.Conv2d(64, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, args['class_num'], kernel_size=1),
            nn.Sigmoid()
        )
        self.wh = nn.Sequential(
            nn.Conv2d(64, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 2, kernel_size=1)
        )
        self.reg = nn.Sequential(
            nn.Conv2d(64, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, 2, kernel_size=1)
        )
        self.init_weights(self.num_layers, pretrianed=True)

    def _make_layer(self, output_channels, num_blocks, stride=1):
        downsample = nn.Sequential(
            nn.Conv2d(self.inter_channels, output_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(output_channels * Bottleneck.expansion, momentum=self.momentum)
        )

        layers = []
        layers.append(Bottleneck(self.inter_channels, output_channels, self.momentum, stride, downsample))
        self.inter_channels = output_channels * Bottleneck.expansion
        for i in range(1, num_blocks):
            layers.append(Bottleneck(self.inter_channels, output_channels))
        return nn.Sequential(*layers)

    def _make_deconv_layer(self):
        layers = []
        output_channels = 256
        for i in range(3):
            layers.append(nn.ConvTranspose2d(self.inter_channels, output_channels, 4, stride=2, padding=1, bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(output_channels, momentum=self.momentum))
            layers.append(nn.ReLU(inplace=True))
            self.inter_channels = output_channels
            output_channels = output_channels // 2
        return nn.Sequential(*layers)

    def forward(self, x, is_train=True):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        result = {}
        result['hm'] = self.hm(x)
        result['wh'] = self.wh(x)
        result['offsets'] = self.reg(x)
        if is_train:
            return result
        

    def init_weights(self, num_layers, pretrianed=True):
        if pretrianed:
            for _, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            for head in self.heads:
                final_layer = getattr(self, head)
                for i, m in enumerate(final_layer.modules()):
                    if isinstance(m, nn.Conv2d):
                        if m.weight.shape[0] == self.heads[head]:
                            if 'hm' in head:
                                nn.init.constant_(m.bias, -2.19)
                            else:
                                nn.init.constant_(m.bias, 0)

            url = model_urls[f'resnet{num_layers}']
            pretrained_state_dict = load_state_dict_from_url(url)

            self.load_state_dict(pretrained_state_dict, strict=False)