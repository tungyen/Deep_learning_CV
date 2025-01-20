import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import math
from dataset import *
from ViT import *


def ViT_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = os.path.join("..", "Dataset", "flowerDataset", "flower_data", "flower_photos")
    weightPath = 'ViT.pth'
    trainImgPaths, trainLabels, valImgPaths, valLabels = dataLoading(path)
    batchSize=4
    
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    
    trainDataset = flowerDataset(trainImgPaths, trainLabels, data_transform["train"])
    valDataset = flowerDataset(valImgPaths, valLabels, data_transform["val"])
    nw = min([os.cpu_count(), batchSize if batchSize > 1 else 0, 8])
    
    trainDataloader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True,
                                 pin_memory=True, num_workers=nw, 
                                 collate_fn=trainDataset.collate_fn)
    
    valDataloader = DataLoader(valDataset, batch_size=batchSize, shuffle=False,
                                 pin_memory=True, num_workers=nw, 
                                 collate_fn=trainDataset.collate_fn)
    
    numEpoch = 10
    bestAcc = 0
    lrf = 0.01
    valNum = len(valDataset)
    
    numClasses = 5
    model = ViT(classNum=numClasses).to(device)
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / numEpoch)) / 2) * (1 - lrf) + lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(opt, lr_lambda=lf)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(numEpoch):
        print("Epoch {} start now!".format(epoch+1))
        # train
        for img, label in tqdm(trainDataloader):
            img, label = img.to(device), label.to(device)
            output = model(img)
            trainLoss = criterion(output, label)
            trainLoss.backward()
            opt.step()  
            opt.zero_grad()
        scheduler.step()
        print("Epoch {}-training loss===>{}".format(epoch+1, trainLoss.item()))
        
        # Validation
        acc = 0.0
        with torch.no_grad():
            for img, label in tqdm(valDataloader):
                output = model(img.to(device))
                predClass = torch.max(output, dim=1)[1]
                acc += torch.eq(predClass, label.to(device)).sum().item()
        valAcc = acc / valNum
        print("Epoch {}-validation Acc===>{}".format(epoch+1, valAcc))
        if valAcc > bestAcc:
            bestAcc = valAcc
            torch.save(model.state_dict(), weightPath)
        
if __name__ =='__main__':
    ViT_train()