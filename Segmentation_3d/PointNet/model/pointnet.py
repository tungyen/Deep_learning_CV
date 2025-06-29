import torch
import torch.nn as nn
import torch.nn.functional as F

class STNkd(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, 64, 1).float()
        self.conv2 = nn.Conv1d(64, 128, 1).float()
        self.conv3 = nn.Conv1d(128, 1024, 1).float()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, out_channels * out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, out_channels))
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        x = x.view(-1, self.out_channels, self.out_channels)
        bias = self.bias.repeat(batch_size, 1, 1)
        x += bias
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pcloud_stn = STNkd(in_channels=in_channels, out_channels=3)
        self.feature_stn = STNkd(in_channels=64, out_channels=64)
        self.conv1 = nn.Conv1d(in_channels, 64, 1).float()
        self.conv2 = nn.Conv1d(64, 128, 1).float()
        self.conv3 = nn.Conv1d(128, 1024, 1).float()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        _, dim, n_points = x.shape
        
        trans1 = self.pcloud_stn(x)
        x = x.transpose(2, 1)
        if dim > 3:
            x, feats = x.split(3, dim=2)
        x = torch.bmm(x, trans1)
        if dim > 3:
            x = torch.cat([x, feats], dim=2)
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        trans2 = self.feature_stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans2)
        x = x.transpose(2,1)
        
        feat = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x_cls = x
        x = x.view(-1, 1024, 1).repeat(1, 1, n_points)
        x_seg = torch.cat([x, feat], 1)
        return x_cls, x_seg, trans1, trans2
    
class SegHead(nn.Module):
    def __init__(self, in_channels, class_num):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 512, 1).float()
        self.conv2 = nn.Conv1d(512, 256, 1).float()
        self.conv3 = nn.Conv1d(256, 128, 1).float()
        self.conv4 = nn.Conv1d(128, class_num, 1).float()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

class ClsHead(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, class_num)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

class PointNetSemseg(nn.Module):
    def __init__(self, class_num, n_feats):
        super().__init__()
        self.encoder = Encoder(n_feats+3)
        self.seg_head = SegHead(1088, class_num)
        
    def forward(self, x):
        _, x_seg, _, _ = self.encoder(x)
        seg_out = self.seg_head(x_seg)
        return seg_out
    
class PointNetPartseg(nn.Module):
    def __init__(self, seg_class_num, cls_class_num, n_feats):
        super().__init__()
        self.encoder = Encoder(n_feats+3)
        self.seg_head = SegHead(1088+cls_class_num, seg_class_num)
        
    def forward(self, x, label):
        _, x_seg, _, _ = self.encoder(x)
        one_hot = F.one_hot(label, num_classes=self.class_num).float()
        one_hot = one_hot.unsqueeze(2).repeat(1, 1, x_seg.shape[2])
        x_seg = torch.cat([x_seg, one_hot], dim=1)
        seg_out = self.seg_head(x_seg)
        return seg_out
    
class PointNetCls(nn.Module):
    def __init__(self, class_num, n_feats):
        super().__init__()
        self.class_num = class_num
        self.encoder = Encoder(n_feats+3)
        self.cls_head = ClsHead(self.class_num)
        
    def forward(self, x):
        x_cls, _, _, _ = self.encoder(x)
        cls_out = self.cls_head(x_cls)
        return cls_out