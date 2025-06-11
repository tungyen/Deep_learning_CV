import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from model.aspp import ASPP, ASPP_Bottleneck

class DeepLabV3(nn.Module):
    def __init__(self, class_num):
        super(DeepLabV3, self).__init__()

        self.class_num = class_num
        # self.model_id = model_id
        # self.project_dir = project_dir
        # self.create_model_dirs()

        self.resnet = ResNet34_OS8()
        self.aspp = ASPP(class_num=class_num) # using ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x)
        output = self.aspp(feature_map)
        output = F.upsample(output, size=(h, w), mode="bilinear")
        return output

    # def create_model_dirs(self):
    #     self.logs_dir = self.project_dir + "/training_logs"
    #     self.model_dir = self.logs_dir + "/model_%s" % self.model_id
    #     self.checkpoints_dir = self.model_dir + "/checkpoints"
    #     if not os.path.exists(self.logs_dir):
    #         os.makedirs(self.logs_dir)
    #     if not os.path.exists(self.model_dir):
    #         os.makedirs(self.model_dir)
    #         os.makedirs(self.checkpoints_dir)