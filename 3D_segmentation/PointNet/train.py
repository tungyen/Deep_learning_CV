import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import *
from pointNet import *
from tqdm import tqdm

class FocalLoss(nn.Module):
    def __init__(self, numClasses=4, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.numClasses = numClasses
        self.gamma = gamma
        
    def forward(self, prediction, annotation):
        loss = nn.CrossEntropyLoss()
        CE_loss = loss(prediction, annotation)
        pt = torch.exp(-CE_loss)
        focalLoss = self.alpha * (1-pt) ** self.gamma * CE_loss
        return focalLoss

def pointNet_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainningPath = "Dataset/train"
    dataset = chairDataset(trainningPath)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    weightPath = 'pointNet.pth'
    
    numClasses = 4
    model = PointNetSegmentation(numClasses).to(device)
    
    opt = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    criterion = FocalLoss()
    
    numEpoch = 10
    for epoch in tqdm(range(numEpoch)):
        for i, (pcd, annotation) in enumerate(dataloader):
            output = model(pcd)
            # output = output.transpose(2, 1)
            trainLoss = criterion(output, annotation)
            opt.zero_grad()
            trainLoss.backward()
            opt.step()
                
            if i % 50 == 0:
                torch.save(model.state_dict(), weightPath)
                
        print("Epoch {}-training loss===>{}".format(epoch, trainLoss.item()))
                
                
if __name__ =='__main__':
    pointNet_train()
    
    
    
    