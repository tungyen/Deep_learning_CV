import torchvision
import torch

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

xy = genfromtxt('../dataset/Diabete/diabetes.csv', delimiter=',', skip_header = 1, dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

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
for epoch in range(maxIteration):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    
    # Update
    optimizer.step()
