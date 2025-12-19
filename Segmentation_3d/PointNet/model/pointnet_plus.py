import torch.nn as nn
import torch.nn.functional as F
import torch

from core.utils import initialize_weights
from Segmentation_3d.PointNet.model.pointnet_plus_layers import PointNetPlusSetAbstraction, PointNetPlusFeaturePropagation

class PointNetPlusBaseModel(nn.Module):
    def __init__(self,
                 n_feats=0,
                 n_samples_list=[1024, 256, 64, 16],
                 radius_list=[0.1, 0.2, 0.4, 0.8],
                 n_points_per_group_list=[32, 32, 32, 32],
                 sa_mlp_out_channels_list=[[32, 48, 64], [64, 96, 128], [128, 196, 256], [256, 384, 512]],
                 fp_mlp_out_channels_list=None):
        super().__init__()
        self.n_samples_list = n_samples_list
        self.sa_layers = nn.ModuleList()
        self.fp_layers = nn.ModuleList() if fp_mlp_out_channels_list is not None else None
        self.sa_feat_channels = [n_feats]
        feat_channel = n_feats
        for i in range(len(n_samples_list)):
            sa_layer = PointNetPlusSetAbstraction(
                n_samples_list[i],
                radius_list[i],
                n_points_per_group_list[i],
                3 + feat_channel,
                sa_mlp_out_channels_list[i]
            )
            self.sa_layers.append(sa_layer)
            feat_channel = sa_layer.out_channels()
            self.sa_feat_channels.append(feat_channel)

        if self.fp_layers is not None:
            for i in range(len(n_points_per_group_list)):
                fp_layer = PointNetPlusFeaturePropagation(
                    in_channels=feat_channel + self.sa_feat_channels[-(i+2)],
                    mlp_out_channels=fp_mlp_out_channels_list[i]
                )
                self.fp_layers.append(fp_layer)
                feat_channel = fp_layer.out_channels()

class PointNetPlusCls(PointNetPlusBaseModel):
    def __init__(self, weight_init, class_num, fc_out_channels, **kwargs):
        super().__init__(**kwargs)
        
        self.fc_layers = nn.ModuleList()
        last_channels = self.sa_layers[-1].out_channels() * self.n_samples_list[-1]
        
        for out_channels in fc_out_channels:
            mlp = nn.ModuleList()
            mlp.append(nn.Linear(last_channels, out_channels))
            mlp.append(nn.BatchNorm1d(out_channels))
            mlp.append(nn.ReLU(inplace=True))
            self.fc_layers.append(mlp)
            last_channels = out_channels
        
        self.cls_head = nn.Linear(last_channels, class_num)
        
    def forward(self, x):
        bs = x.shape[0]
        x = x.transpose(1, 2).contiguous()
        cur_feats = None
        for i, layer in enumerate(self.sa_layers):
            x, cur_feats = layer(x, cur_feats)

        feats = cur_feats.view([bs, -1])
        for mlp in self.fc_layers:
            for block in mlp:
                feats = block(feats)
            feats = F.dropout(feats, p=0.5, training=self.training)
            
        cls_outs = self.cls_head(feats)
        return cls_outs, None

class PointNetPlusSemseg(PointNetPlusBaseModel):
    def __init__(self, weight_init, class_num, **kwargs):
        super().__init__(**kwargs)
        
        feat_channel = self.fp_layers[-1].out_channels()
        self.seg_head = nn.Conv1d(feat_channel, class_num, 1)
        if weight_init is not None:
            self.apply(initialize_weights(weight_init))
        
    def forward(self, x):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        feats = x[:, 3:, :].transpose(1, 2).contiguous()
        
        xyzs = [xyz]
        feats_list = [feats]
        for sa_layer in self.sa_layers:
            xyz, feats = sa_layer(xyz, feats)
            xyzs.append(xyz)
            feats_list.append(feats)

        for i, fp_layer in enumerate(self.fp_layers):
            feats = fp_layer(xyzs[-(i+1)], feats, xyzs[-(i+2)], feats_list[-(i+2)], k=3)
        
        seg_outs = self.seg_head(feats.transpose(1, 2))
        return seg_outs, None
    
class PointNetPlusPartseg(PointNetPlusBaseModel):
    def __init__(self, seg_class_num, cls_class_num, weight_init, **kwargs):
        super().__init__(**kwargs)
        self.cls_class_num = cls_class_num
        self.seg_head = nn.Conv1d(self.fp_layers[-1].out_channels() + cls_class_num, seg_class_num, 1)
        
    def forward(self, x, cls_label):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        feats = x[:, 3:, :].transpose(1, 2).contiguous()
        xyzs = [xyz]
        feats_list = [feats]

        for sa_layer in self.sa_layers:
            xyz, feats = sa_layer(xyz, feats)
            xyzs.append(xyz)
            feats_list.append(feats)
        for i, fp_layer in enumerate(self.fp_layers):
            feats = fp_layer(xyzs[-(i+1)], feats, xyzs[-(i+2)], feats_list[-(i+2)], k=3)

        one_hot = F.one_hot(cls_label, num_classes=self.cls_class_num).float()
        one_hot = one_hot.unsqueeze(1).repeat(1, feats.shape[1], 1)
        feats = torch.cat([feats, one_hot], dim=2)
        
        seg_outs = self.seg_head(feats.transpose(1, 2))
        return seg_outs, None

    def post_process(self, outputs, cls_labels, class_dict):
        instance2parts, _, label2class = class_dict
        pred_classes = torch.zeros((outputs.shape[0], outputs.shape[2]))
        for i in range(outputs.shape[0]):
            instance = label2class[cls_labels[i].item()]
            logits = outputs[i, :, :].cpu()
            pred_classes[i, :] = torch.argmax(logits[instance2parts[instance], :], 0) + instance2parts[instance][0]
        return pred_classes