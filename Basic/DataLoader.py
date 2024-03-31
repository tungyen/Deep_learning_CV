import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

class DiabetesDataset(Dataset):
    
    def __init__(self, filepath):
        xy = genfromtxt(filepath, delimiter=',', skip_header = 1, dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len

class LogisticRegressionModelMD(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModelMD, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
    
model = LogisticRegressionModelMD()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
maxIteration = 1000


if __name__ == '__main__':
    dataset = DiabetesDataset('../dataset/Diabete/diabetes.csv')
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)
    for epoch in range(maxIteration):
        for i, data in enumerate(train_loader, 0):
                # Prepare data
                inputs, labels = data
                
                # Forward
                y_pred = model(inputs)
                loss = criterion(y_pred, labels)
                print(epoch, i, loss.item())
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Update
                optimizer.step()