import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class trainTitanicDataset(Dataset):
    
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        labels = ["Survived"]
        features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
        data["Age"] = data["Age"].fillna(data["Age"].mean())
        self.len = data.shape[0]
        self.train_data = data[:int(self.len*0.8)]
        
        # pd.get_dummies is used to generate one-hot code
        self.x_data = torch.from_numpy(np.array(pd.get_dummies(self.train_data[features])).astype(np.float32))
        self.y_data = torch.from_numpy(np.array(self.train_data[labels]).astype(np.float32))
        self.train_len = self.train_data.shape[0]
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.train_len
    
train_dataset = trainTitanicDataset('../dataset/titanic/train.csv')
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers = 0)

class devTitanicDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv(filepath)
        labels = ["Survived"]
        features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
        data["Age"] = data["Age"].fillna(data["Age"].mean())
        self.len = data.shape[0]
        self.dev_data = data[int(self.len*0.8):]
        
        # pd.get_dummies is used to generate one-hot code
        self.x_data = torch.from_numpy(np.array(pd.get_dummies(self.dev_data[features])).astype(np.float32))
        self.y_data = torch.from_numpy(np.array(self.dev_data[labels]).astype(np.float32))
        self.dev_len = self.dev_data.shape[0]
        
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.dev_len
    
dev_dataset = devTitanicDataset('../dataset/titanic/train.csv')
dev_loader = DataLoader(dataset=dev_dataset, batch_size=8, shuffle=False, num_workers = 0)

class Model(torch.nn.Module):
    
    def __init__(self):
        
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(7, 6)
        self.linear2 = torch.nn.Linear(6, 6)
        self.linear3 = torch.nn.Linear(6, 3)
        self.linear4 = torch.nn.Linear(3, 2)
        self.linear5 = torch.nn.Linear(2, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.relu(self.linear4(x))
        x = self.sigmoid(self.linear5(x))
        return x
    
model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=0.01, 
                             betas=(0.9, 0.999), 
                             eps=1e-8, 
                             weight_decay=0, 
                             amsgrad=False)

def train(epoch):
    trainLoss = 0.0
    count = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        trainLoss += loss.item() + 1e-8
        count = i
    # print('epoch: ', epoch+1, 'train_loss: ', trainLoss/count, end = ',')
    
def dev():
    correct = 0.0
    total = 0.0
    dev_mean_loss = 0.0
    for i, data in enumerate(dev_loader, 0):
        inputs, labels = data
        output = model(inputs)
        devLoss = criterion(output, labels)
        dev_mean_loss += devLoss.item() + 1e-8
        total += labels.size(0)
        correct += np.sum((np.round(output.detach().numpy()) == labels))
        acc = correct / total
        count = i

    print('dev loss: ', dev_mean_loss/count, 'Accuracy on dev set: ', acc)
    
    
if __name__ == '__main__':
    
    for epoch in range(100):
        train(epoch)
        dev()
        
    test_data = pd.read_csv('../dataset/titanic/test.csv')
    features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Age"]
    test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())
    test = torch.from_numpy(np.array(pd.get_dummies(test_data[features])).astype(np.float32))
    
    with torch.no_grad():
        y_pred = model(test)
        y = []
        for i in y_pred:
            if i >= 0.5:
                y.append(1)
            else:
                y.append(0)
        output = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived': y})
        output.to_csv('../dataset/titanic/predict.csv', index=False)