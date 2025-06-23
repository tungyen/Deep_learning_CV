import torch
import torch.nn as nn
import torch.nn.functional as F

# define the model structure
class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1).float()
        self.conv2 = nn.Conv1d(64, 128, 1).float()
        self.conv3 = nn.Conv1d(128, 1024, 1).float()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bias = nn.Parameter(torch.zeros(1, 3, 3))
        
    def forward(self, x):
        batchNumber = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        x = x.view(-1, 3, 3)
        bias = self.bias.repeat(batchNumber, 1, 1)
        x += bias
        return x
    
# define the model structure
class STNkd(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1).float()
        self.conv2 = nn.Conv1d(64, 128, 1).float()
        self.conv3 = nn.Conv1d(128, 1024, 1).float()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bias = nn.Parameter(torch.zeros(1, k, k))
        self.k = k
        
    def forward(self, x):
        batchNumber = x.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        x = x.view(-1, self.k, self.k)
        bias = self.bias.repeat(batchNumber, 1, 1)
        x += bias
        return x
    
    
class Encoder(nn.Module):
    def __init__(self, dimension=3):
        super().__init__()
        self.pointCloudSTN = STN3d()
        self.featureSTN = STNkd()
        self.conv1 = nn.Conv1d(dimension, 64, 1).float()
        self.conv2 = nn.Conv1d(64, 128, 1).float()
        self.conv3 = nn.Conv1d(128, 1024, 1).float()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.dim = dimension
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        n = x.shape[2]
        
        trans1 = self.pointCloudSTN(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans1)
        x = x.transpose(2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        
        trans2 = self.featureSTN(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans2)
        x = x.transpose(2,1)
        
        feat = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x_cls = x
        x = x.view(-1, 1024, 1).repeat(1, 1, n)
        x_seg = torch.cat([x, feat], 1)
        return x_cls, x_seg, trans1, trans2
    
class SegHead(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.numClasses = class_num
        self.conv1 = nn.Conv1d(1088, 512, 1).float()
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
        self.class_num = class_num
        
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
    
    
class PointNetSeg(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        self.encoder = Encoder()
        self.seg_head = SegHead(self.class_num)
        
    def forward(self, x):
        _, x_seg, _, _ = self.encoder(x)
        seg_out = self.seg_head(x_seg)
        return seg_out
    
class PointNetCls(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self.class_num = class_num
        self.encoder = Encoder()
        self.cls_head = ClsHead(self.class_num)
        
    def forward(self, x):
        x_cls, _, _, _ = self.encoder(x)
        cls_out = self.cls_head(x_cls)
        return cls_out