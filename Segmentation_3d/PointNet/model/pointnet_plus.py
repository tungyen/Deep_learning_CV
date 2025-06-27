import torch.nn as nn
import torch.nn.functional as F
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
        return cls_outs

class PointNetPlusSeg(nn.Module):
    def __init__(self, class_num, n_feats, pointnet_plus_seg_dict):
        super().__init__()
        
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
        
        self.seg_head = nn.Conv1d(self.fp4.out_channels(), class_num, 1)
        
    def forward(self, x):
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
        
        seg_outs = self.seg_head(f4_feats.transpose(1, 2))
        return seg_outs