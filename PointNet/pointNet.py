import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# define the model structure
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
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
        super(STNkd, self).__init__()
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
    
    
class PointNetEncoder(nn.Module):
    def __init__(self, dimension=3):
        super(PointNetEncoder, self).__init__()
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
        xCLS = x
        x = x.view(-1, 1024, 1).repeat(1, 1, n)
        xSEG = torch.cat([x, feat], 1)
        return xCLS, xSEG, trans1, trans2
    
class SegmentationTask(nn.Module):
    def __init__(self, numClasses):
        super(SegmentationTask, self).__init__()
        self.numClasses = numClasses
        self.conv1 = nn.Conv1d(1088, 512, 1).float()
        self.conv2 = nn.Conv1d(512, 256, 1).float()
        self.conv3 = nn.Conv1d(256, 128, 1).float()
        self.conv4 = nn.Conv1d(128, numClasses, 1).float()
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        batchNumber = x.shape[0]
        n = x.shape[2]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        
        # x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.numClasses), dim=-1)
        # x = x.view(batchNumber, n, self.numClasses)
        return x
    
    
class PointNetSegmentation(nn.Module):
    def __init__(self, numClasses=4):
        super(PointNetSegmentation, self).__init__()
        self.numClasses = numClasses
        self.encoder = PointNetEncoder()
        self.segmentationTask = SegmentationTask(self.numClasses)
        
    def forward(self, x):
        xCLS, xSEG, trans1, trans2 = self.encoder(x)
        segmentationOuput = self.segmentationTask(xSEG)
        return segmentationOuput
    
# if __name__ == '__main__':
    
#     model = PointNetSegmentation()
#     x = torch.randn(5, 3, 1500)
#     print(x.shape)
#     output = model(x)
#     print(output.shape)