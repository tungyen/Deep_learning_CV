import numpy as np
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree

class VOCDataset(Dataset):
    
    def __init__(self, rootPath, transform=None, name='train.txt'):
        self.rootPath = rootPath
        self.transform = transform
        self.name = name
        self.imgPath = os.path.join(self.rootPath, 'VOC2012', 'JPEGImages')
        self.annotationPath = os.path.join(self.rootPath, 'VOC2012', 'Annotations')
        
        txtPath = os.path.join(self.rootPath, 'VOC2012', 'ImageSets', 'Main', self.name)
        with open(txtPath) as read:
            xmlList = [os.path.join(self.annotationPath, line.strip()+'.xml')
                       for line in read.readlines() if len(line.strip()) > 0]

        # Filtering out annotations without objects or not existed
        self.xmlList = []
        for xmlPath in xmlList:
            if os.path.exists(xmlPath) is False:
                continue
            
            with open(xmlPath) as file:
                xmlStr = file.read()
            xmlFile = etree.fromstring(xmlStr)
            xmlContent = self.parseXmlToDict(xmlFile)['annotation']
            if 'object' not in xmlContent:
                continue
            self.xmlList.append(xmlPath)
            
        jsonFilePath = os.path.join(self.rootPath, "pascal_voc_classes.json")
        with open(jsonFilePath) as fid:
            self.classDict = json.load(fid)
            
    def __len__(self):
        return len(self.xmlList)
    
    
    def __getitem__(self, idx):
        xmlPath = self.xmlList[idx]
        with open(xmlPath) as fid:
            xmlStr = fid.read()
        xmlFile = etree.fromstring(xmlStr)
        xmlContent = self.parseXmlToDict(xmlFile)['annotation']
        height = int(xmlContent["size"]["height"])
        width = int(xmlContent["size"]["width"])
        imgPath = os.path.join(self.imgPath, xmlContent['filename'])
        img = Image.open(imgPath)
        
        # Parsing the XML file
        bboxes = []
        labels = []
        iscrowd = []
        for obj in xmlContent['object']:
            xMin = float(obj['bndbox']['xmin'])
            xMax = float(obj['bndbox']['xmax'])
            yMin = float(obj['bndbox']['ymin'])
            yMax = float(obj['bndbox']['ymax'])
            
            if xMax <= xMin or yMax <= yMin:
                continue
            
            bboxes.append([xMin, yMin, xMax, yMax])
            labels.append(self.classDict[obj['name']])
            if 'difficult' in obj:
                iscrowd.append(int(obj['difficult']))
            else:
                iscrowd.append(0)
                
        # Transforming all data to tensor
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        imgID = torch.as_tensor([idx])
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        
        annotation = {}
        annotation['bboxes'] = bboxes
        annotation['labels'] = labels
        annotation['iscrowd'] = iscrowd
        annotation['imgID'] = imgID
        annotation['area'] = area
        
        if self.transform is not None:
            img, annotation = self.transform(img, annotation)
        return img, height, width, annotation
        
                
    def parseXmlToDict(self, xml):
        # Inputs:
        #     xml - xml tree got from lxml.tree
        # Outputs:
        #     xmlDict - Dictionary format of the input xml tree
        if len(xml) == 0:
            xmlDict = {xml.tag: xml.text}
            return xmlDict
        
        res = {}
        for child in xml:
            childRes = self.parseXmlToDict(child)
            if child.tag != 'object':
                res[child.tag] = childRes[child.tag]
            else:
                if child.tag not in res:
                    res[child.tag] = []
                res[child.tag].append(childRes[child.tag])
        xmlDict = {xml.tag: res}
        return xmlDict
    
    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
    
from transforms import *
from draw_box_utils import draw_objs
import matplotlib.pyplot as plt
import torchvision.transforms as ts
import random

# read class_indict
category_index = {}
try:
    json_file = open('../../Dataset/VOCdevkit/pascal_voc_classes.json', 'r')
    class_dict = json.load(json_file)
    category_index = {str(v): str(k) for k, v in class_dict.items()}
except Exception as e:
    print(e)
    exit(-1)

data_transform = {
    "train": Compose([ToTensor(), RandomHorizontalFlip(0.5)]),
    "val": Compose([ToTensor()])
}

# load train data set
root = "../../Dataset/VOCdevkit"
train_data_set = VOCDataset(root, data_transform["train"], "train.txt")
# train_data_set = resizeImageLabel(800, 1333, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

for index in random.sample(range(0, len(train_data_set)), k=5):
    img, h, w, target = train_data_set[index]
    print(target['bboxes'])
    img = ts.ToPILImage()(img)
    plot_img = draw_objs(img,
                         target["bboxes"].numpy(),
                         target["labels"].numpy(),
                         np.ones(target["labels"].shape[0]),
                         category_index=category_index,
                         box_thresh=0.5,
                         line_thickness=3,
                         font='arial.ttf',
                         font_size=20)
    plt.imshow(plot_img)
    plt.show()
            