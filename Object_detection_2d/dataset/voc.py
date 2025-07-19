from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import os
import tarfile
import torch
import xml.etree.ElementTree as ET

voc_class2id = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20,
}

voc_id2class = {v:k for k, v in voc_class2id.items()}

VOC_DATASET_YEAR_DICT = {
    '2012': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '6cd6e144f989b92b3379bac3b3de84fd',
        'base_dir': 'VOCdevkit/VOC2012'
    },
    '2011': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2011/VOCtrainval_25-May-2011.tar',
        'filename': 'VOCtrainval_25-May-2011.tar',
        'md5': '6c3384ef61512963050cb5d687e5bf1e',
        'base_dir': 'TrainVal/VOCdevkit/VOC2011'
    },
    '2010': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar',
        'filename': 'VOCtrainval_03-May-2010.tar',
        'md5': 'da459979d0c395079b5c75ee67908abb',
        'base_dir': 'VOCdevkit/VOC2010'
    },
    '2009': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar',
        'filename': 'VOCtrainval_11-May-2009.tar',
        'md5': '59065e4b188729180974ef6572f6a212',
        'base_dir': 'VOCdevkit/VOC2009'
    },
    '2008': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2008/VOCtrainval_14-Jul-2008.tar',
        'filename': 'VOCtrainval_11-May-2012.tar',
        'md5': '2629fa636546599198acfcfbfcf1904a',
        'base_dir': 'VOCdevkit/VOC2008'
    },
    '2007': {
        'url': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'filename': 'VOCtrainval_06-Nov-2007.tar',
        'md5': 'c52e279531787c972589f7e41ab4ae64',
        'base_dir': 'VOCdevkit/VOC2007'
    }
}

def download_extract(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)

class VocDetectionDataset(Dataset):
    def __init__(self, root, year='2012', split='train', download=False, transform=None, keep_diffucult=False):
        self.root = os.path.expanduser(root)
        self.year = year
        self.url = VOC_DATASET_YEAR_DICT[year]['url']
        self.filename = VOC_DATASET_YEAR_DICT[year]['filename']
        self.md5 = VOC_DATASET_YEAR_DICT[year]['md5']
        self.transform = transform
        self.split = split
        self.keep_difficult = keep_diffucult
        
        base_dir = VOC_DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        
        if download:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')
        
        img_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, "Annotations")
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_file = os.path.join(splits_dir, f'{split}.txt')
            
        if not os.path.exists(split_file):
            raise ValueError(
                'Wrong image_set entered! Please use split=train or split=trainval or split=val')

        with open(os.path.join(split_file), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(img_dir, f"{x}.jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, f"{x}.xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        ann_path = self.annotations[index]
        
        boxes = []
        labels= []
        
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        for obj in root.findall("object"):
            difficult = int(obj.find("difficult").text == "1")
            if (self.keep_difficult is False) and difficult == 1:
                continue
            name = obj.find("name").text.lower().strip()
            label = voc_class2id.get(name)
            if label is None:
                continue
            
            box = obj.find("bndbox")
            xmin = int(box.find("xmin").text) - 1
            xmax = int(box.find("xmax").text) - 1
            ymin = int(box.find("ymin").text) - 1
            ymax = int(box.find("ymax").text) - 1
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            "bboxes": boxes,
            "labels": labels,
            "image_id":torch.tensor([index])
        }
        
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, target

    def __len__(self):
        return len(self.images)
    
    def collate_fn(self, batch):
        imgs = list()
        targets = list()
        for img, target in batch:
            imgs.append(img)
            targets.append(target)

        images = torch.stack(images, dim=0)
        return images, targets
    
    
