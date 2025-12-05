import torch.nn as nn
import torch.nn.functional as F
import torch

from Segmentation_3d.utils import initialize_weights
from Segmentation_3d.PointNet.model.pointnet_plus_layers import PointNetPlusSetAbstraction, PointNetPlusFeaturePropagation

class PointNetPlusCls(nn.Module):
    def __init__(self, class_num, pointnet_plus_cls_dict, fc_out_channels=[512, 256]):
        super().__init__()
        
        self.sa1 = PointNetPlusSetAbstraction(
            pointnet_plus_cls_dict['n_samples_list'][0],
            pointnet_plus_cls_dict['radius_list'][0],
            pointnet_plus_cls_dict['n_points_per_group_list'][0], 3,
            pointnet_plus_cls_dict['sa_mlp_out_channels_list'][0])
        
        self.sa2 = PointNetPlusSetAbstraction(
            pointnet_plus_cls_dict['n_samples_list'][1],
            pointnet_plus_cls_dict['radius_list'][1],
            pointnet_plus_cls_dict['n_points_per_group_list'][1],
            3+self.sa1.out_channels(),
            pointnet_plus_cls_dict['sa_mlp_out_channels_list'][1])
        
        self.sa3 = PointNetPlusSetAbstraction(
            pointnet_plus_cls_dict['n_samples_list'][2],
            pointnet_plus_cls_dict['radius_list'][2],
            pointnet_plus_cls_dict['n_points_per_group_list'][2],
            3+self.sa2.out_channels(),
            pointnet_plus_cls_dict['sa_mlp_out_channels_list'][2])
        
        self.fc_layers = nn.ModuleList()
        last_channels = self.sa3.out_channels() * pointnet_plus_cls_dict['n_samples_list'][2]
        
        for out_channels in fc_out_channels:
            mlp = nn.ModuleList()
            mlp.append(nn.Linear(last_channels, out_channels))
            mlp.append(nn.BatchNorm1d(out_channels))
            mlp.append(nn.ReLU(inplace=True))
            self.fc_layers.append(mlp)
            last_channels = out_channels
        
        self.cls_head = nn.Linear(last_channels, class_num)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.transpose(1, 2).contiguous()
        
        sa1_xyz, sa1_feats = self.sa1(x)
        sa2_xyz, sa2_feats = self.sa2(sa1_xyz, sa1_feats)
        _, sa3_feats = self.sa3(sa2_xyz, sa2_feats)
        feats = sa3_feats.view([batch_size, -1])
        for mlp in self.fc_layers:
            for block in mlp:
                feats = block(feats)
            feats = F.dropout(feats, p=0.5, training=self.training)
            
        cls_outs = self.cls_head(feats)
        return cls_outs, None

class PointNetPlusSemseg(nn.Module):
    def __init__(self, class_num=4,
                 n_feats=0,
                 n_samples_list=[1024, 256, 64, 16],
                 radius_list=[0.1, 0.2, 0.4, 0.8],
                 n_points_per_group_list=[32, 32, 32, 32],
                 sa_mlp_out_channels_list=[[32, 48, 64], [64, 96, 128], [128, 196, 256], [256, 384, 512]],
                 fp_mlp_out_channels_list=[[1024, 512], [512, 512], [512, 256], [256, 256, 128]],
                 weight_init="xavier"
        ):
        super().__init__()

        self.sa_layers = nn.ModuleList()
        self.fp_layers = nn.ModuleList()
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

        for i in range(len(n_points_per_group_list)):
            fp_layer = PointNetPlusFeaturePropagation(
                in_channels=feat_channel + self.sa_feat_channels[-(i+2)],
                mlp_out_channels=fp_mlp_out_channels_list[i]
            )
            self.fp_layers.append(fp_layer)
            feat_channel = fp_layer.out_channels()
        
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
    
class PointNetPlusPartseg(nn.Module):
    def __init__(self, seg_class_num, cls_class_num, n_feats, pointnet_plus_seg_dict):
        super().__init__()
        self.cls_class_num = cls_class_num
        self.sa1 = PointNetPlusSetAbstraction(
            pointnet_plus_seg_dict['n_samples_list'][0],
            pointnet_plus_seg_dict['radius_list'][0],
            pointnet_plus_seg_dict['n_points_per_group_list'][0],
            3 + n_feats,
            pointnet_plus_seg_dict['sa_mlp_out_channels_list'][0])
        
        self.sa2 = PointNetPlusSetAbstraction(
            pointnet_plus_seg_dict['n_samples_list'][1],
            pointnet_plus_seg_dict['radius_list'][1],
            pointnet_plus_seg_dict['n_points_per_group_list'][1],
            3+self.sa1.out_channels(),
            pointnet_plus_seg_dict['sa_mlp_out_channels_list'][1])
        
        self.sa3 = PointNetPlusSetAbstraction(
            pointnet_plus_seg_dict['n_samples_list'][2],
            pointnet_plus_seg_dict['radius_list'][2],
            pointnet_plus_seg_dict['n_points_per_group_list'][2],
            3+self.sa2.out_channels(),
            pointnet_plus_seg_dict['sa_mlp_out_channels_list'][2])
        
        self.sa4 = PointNetPlusSetAbstraction(
            pointnet_plus_seg_dict['n_samples_list'][3],
            pointnet_plus_seg_dict['radius_list'][3],
            pointnet_plus_seg_dict['n_points_per_group_list'][3],
            3+self.sa3.out_channels(),
            pointnet_plus_seg_dict['sa_mlp_out_channels_list'][3])
        
        self.fp1 = PointNetPlusFeaturePropagation(
            in_channels=self.sa4.out_channels() + self.sa3.out_channels(),
            mlp_out_channels=pointnet_plus_seg_dict['fp_mlp_out_channels_list'][0])
        
        self.fp2 = PointNetPlusFeaturePropagation(
            in_channels=self.fp1.out_channels() + self.sa2.out_channels(),
            mlp_out_channels=pointnet_plus_seg_dict['fp_mlp_out_channels_list'][1])
        
        self.fp3 = PointNetPlusFeaturePropagation(
            in_channels=self.fp2.out_channels() + self.sa1.out_channels(),
            mlp_out_channels=pointnet_plus_seg_dict['fp_mlp_out_channels_list'][2])
        
        self.fp4 = PointNetPlusFeaturePropagation(
            in_channels=self.fp3.out_channels() + n_feats,
            mlp_out_channels=pointnet_plus_seg_dict['fp_mlp_out_channels_list'][3])
        
        self.seg_head = nn.Conv1d(self.fp4.out_channels() + cls_class_num, seg_class_num, 1)
        
    def forward(self, x, cls_label):
        xyz = x[:, :3, :].transpose(1, 2).contiguous()
        feats = x[:, 3:, :].transpose(1, 2).contiguous()
        
        sa1_xyz, sa1_feats = self.sa1(xyz, feats)
        sa2_xyz, sa2_feats = self.sa2(sa1_xyz, sa1_feats)
        sa3_xyz, sa3_feats = self.sa3(sa2_xyz, sa2_feats)
        sa4_xyz, sa4_feats = self.sa4(sa3_xyz, sa3_feats)
        
        f1_feats = self.fp1(sa4_xyz, sa4_feats, sa3_xyz, sa3_feats, k=3)
        f2_feats = self.fp2(sa3_xyz, f1_feats, sa2_xyz, sa2_feats, k=3)
        f3_feats = self.fp3(sa2_xyz, f2_feats, sa1_xyz, sa1_feats, k=3)
        f4_feats = self.fp4(sa1_xyz, f3_feats, xyz, feats, k=3)
        one_hot = F.one_hot(cls_label, num_classes=self.cls_class_num).float()
        one_hot = one_hot.unsqueeze(1).repeat(1, f4_feats.shape[1], 1)
        f4_feats = torch.cat([f4_feats, one_hot], dim=2)
        
        seg_outs = self.seg_head(f4_feats.transpose(1, 2))
        return seg_outs, None

    def post_process(self, outputs, cls_labels, class_dict):
        instance2parts, _, label2class = class_dict
        pred_classes = torch.zeros((outputs.shape[0], outputs.shape[2]))
        for i in range(outputs.shape[0]):
            instance = label2class[cls_labels[i].item()]
            logits = outputs[i, :, :].cpu()
            pred_classes[i, :] = torch.argmax(logits[instance2parts[instance], :], 0) + instance2parts[instance][0]
        return pred_classes