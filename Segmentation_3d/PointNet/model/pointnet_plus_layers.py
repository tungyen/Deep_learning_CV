import torch
import torch.nn as nn
import torch.nn.functional as F

from Segmentation_3d.PointNet.model.utils import ball_query, furthest_point_sampling, k_nearest_neighbor, batch_indexing

def PointNetPlusSetAbstraction(n_samples, radius, n_points_per_group, in_channels, mlp_out_channels):
    assert isinstance(n_samples, int)
    if isinstance(radius, list):
        assert n_samples > 1
        assert isinstance(n_points_per_group, list)
        assert isinstance(mlp_out_channels, list)
        assert len(radius) == len(n_points_per_group) == len(mlp_out_channels)
        return PointNetPlusSetAbstractionMSG(n_samples, radius, n_points_per_group, in_channels, mlp_out_channels)
    else:
        return PointNetPlusSetAbstractionSSG(n_samples, radius, n_points_per_group, in_channels, mlp_out_channels)

class PointNetPlusSetAbstractionSSG(nn.Module):
    def __init__(self, n_samples, radius, n_points_per_group, in_channels, mlp_out_channels):
        super().__init__()
        self.n_samples = n_samples
        self.radius = radius
        self.n_points_per_group = n_points_per_group
        self.mlp_out_channels = mlp_out_channels
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        for out_channels in mlp_out_channels:
            self.mlp_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            
    def forward(self, points_xyz, features=None):
        batch_size, n_points = points_xyz.shape[0], points_xyz.shape[1]
        
        if features is None:
            features = torch.zeros([batch_size, n_points, 0], device=points_xyz.device)
            
        if self.n_samples > 1:
            centroid_indexes = furthest_point_sampling(points_xyz, self.n_samples)
            centroids = batch_indexing(points_xyz, centroid_indexes)
            group_indexes = ball_query(points_xyz, centroids, self.radius, self.n_points_per_group)
            group_xyz = batch_indexing(points_xyz, group_indexes)
            group_features = batch_indexing(features, group_indexes)
            group_xyz_norm = group_xyz - centroids.view(batch_size, self.n_samples, 1, 3)
            group_features = torch.cat([group_xyz_norm, group_features], dim=-1)
        else:
            centroids = torch.zeros([batch_size, 1, 3], dtype=points_xyz.dtype, device=points_xyz.device)
            group_xyz = points_xyz.view(batch_size, 1, n_points, 3)
            group_features = torch.cat([group_xyz, features.view(batch_size, 1, n_points, -1)], dim=-1)
        
        # PointNet 
        group_features = group_features.transpose(1, 3)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            group_features = F.relu(bn(conv(group_features)))
        group_features = torch.max(group_features, 2)[0]
        group_features = group_features.transpose(1, 2)
        
        return centroids, group_features
    
    def out_channels(self):
        return self.mlp_out_channels[-1]
    
class PointNetPlusSetAbstractionMSG(nn.Module):
    def __init__(self, n_samples, radius_list, n_points_per_group_list, in_channels, mlp_out_channels_list):
        super().__init__()
        self.n_samples = n_samples
        self.radius_list = radius_list
        self.n_points_per_group_list = n_points_per_group_list
        self.mlp_out_channels_list = mlp_out_channels_list
        
        self.mlp_convs_list = nn.ModuleList()
        self.mlp_bns_list = nn.ModuleList()
        
        for mlp_out_channels in mlp_out_channels_list:
            mlp_convs = nn.ModuleList()
            mlp_bns = nn.ModuleList()
            last_channels = in_channels
            for out_channels in mlp_out_channels:
                mlp_convs.append(nn.Conv2d(last_channels, out_channels, 1))
                mlp_bns.append(nn.BatchNorm2d(out_channels))
                last_channels = out_channels
            self.mlp_convs_list.append(mlp_convs)
            self.mlp_bns_list.append(mlp_bns)
            
    def forward(self, points_xyz, features=None):
        batch_size, n_points = points_xyz.shape[0], points_xyz.shape[1]
        
        if features is None:
            features = torch.zeros([batch_size, n_points, 0], device=points_xyz.device)
        
        centroids_indexes = furthest_point_sampling(points_xyz, self.n_samples)
        centroids = batch_indexing(points_xyz, centroids_indexes)
        
        multi_scale_features = []
        for radius, n_points_per_group, mlp_convs, mlp_bns in zip(
            self.radius_list, self.n_points_per_group_list, self.mlp_convs_list, self.mlp_bns_list):
            
            group_indexes = ball_query(points_xyz, centroids, radius, n_points_per_group)
            group_xyz = batch_indexing(points_xyz, group_indexes)
            group_features = batch_indexing(features, group_indexes)
            group_xyz_norm = group_xyz - centroids.view(batch_size, self.n_samples, 1, 3)
            group_features = torch.cat([group_xyz_norm, group_features], dim=-1)
            
            group_features = group_features.transpose(1, 3)
            for conv, bn in zip(mlp_convs, mlp_bns):
                group_features = F.relu(bn(conv(group_features)))
            group_features = torch.max(group_features, 2)[0]
            group_features = group_features.transpose(1, 2)
            
            multi_scale_features.append(group_features)
            
        multi_scale_features = torch.cat(multi_scale_features, dim=-1)
        return centroids, multi_scale_features
    
    def out_channels(self):
        return sum(mlp_out_channels[-1] for mlp_out_channels in self.mlp_out_channels_list)
    
class PointNetPlusFeaturePropagation(nn.Module):
    def __init__(self, in_channels, mlp_out_channels):
        super().__init__()
        self.mlp_out_channels = mlp_out_channels
        
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        for out_channels in mlp_out_channels:
            self.mlp_convs.append(nn.Conv1d(in_channels, out_channels, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels
            
    def forward(self, sampled_xyz, sampled_features, original_xyz, original_features, k):
        # For each original xyz point in original xyz, take the top k closest sample point from sampled_xyz
        knn_dists, knn_indexes = k_nearest_neighbor(sampled_xyz, original_xyz, k)
        knn_weights = 1.0 / (knn_dists + 1e-8)
        knn_weights = knn_weights / torch.sum(knn_weights, dim=-1, keepdim=True)
        
        knn_features = batch_indexing(sampled_features, knn_indexes)
        interpolated_features = torch.sum(knn_features * knn_weights[:, :, :, None], dim=2)
        concatenated_features = torch.cat([original_features, interpolated_features], dim=-1)
        
        concatenated_features = concatenated_features.transpose(1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            concatenated_features = F.relu(bn(conv(concatenated_features)))
        concatenated_features = concatenated_features.transpose(1, 2)

        return concatenated_features
    
    def out_channels(self):
        return self.mlp_out_channels[-1]