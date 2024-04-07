from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from Unet import *
from torchvision.utils import save_image


def unet_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weightPath = 'unet.pth'
    dataPath = '../Dataset/VOCdevkit/VOC2012'
    savePath = 'save/'

    dataLoader = DataLoader(vocDataset(dataPath), batch_size=4, shuffle=True)
    unet = UNET().to(device)
    if os.path.exists(weightPath):
        unet.load_state_dict(torch.load(weightPath))
        print("Successfully loading weight!")
    else:
        print("Not successfully loading weight!")
        
    opt = optim.Adam(unet.parameters())
    loss = nn.CrossEntropyLoss()
    
    maxEpoch = 3
    for epoch in range(maxEpoch):
        for i, (img, annotation) in enumerate(dataLoader):
            img, annotation = img.to(device), annotation.to(device)
            outputImg = unet(img)
            trainLoss = loss(outputImg, annotation)
            opt.zero_grad()
            trainLoss.backward()
            opt.step()
            
            if i % 50 == 0:
                torch.save(unet.state_dict(), weightPath)
            
        print("Epoch {}-training loss===>{}".format(epoch, trainLoss.item()))
                
            
                
#             _img = img[0]
#             s = img.shape[-1]
#             __annotation = annotation[0].view()
#             _outputImg = outputImg[0]
            
#             stacking = torch.stack([_img, __annotation, _outputImg], dim=0)
#             save_image(stacking, savePath+str(i)+'.png')
            
        epoch += 1
        
if __name__ == '__main__':
    unet_train()
            