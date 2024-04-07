from torch.utils.data import Dataset
from torchvision import transforms
import os
from utils import *

transform = transforms.Compose([
    transforms.ToTensor()
])

class vocDataset(Dataset):
    
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.annotationPath = os.listdir(os.path.join(dataPath, 'SegmentationClass'))
        
    def __len__(self):
        return len(self.annotationPath)
    
    def __getitem__(self, index):
        segmentName = self.annotationPath[index] # xx.png
        segmentPath = os.path.join(self.dataPath, 'SegmentationClass', segmentName)
        imagePath = os.path.join(self.dataPath, 'JPEGImages', segmentName.replace('png', 'jpg'))
        
        segmentImg = maskResize(segmentPath)
        img = imageResize(imagePath)
        s = segmentImg.size[-1]
        return transform(img), transform(segmentImg).view(s, s).long()
    
if __name__ == '__main__':
    dataPath = '../Dataset/VOCdevkit/VOC2012'
    data = vocDataset(dataPath)
    img1, label1 = data[0]
    print(label1.shape)