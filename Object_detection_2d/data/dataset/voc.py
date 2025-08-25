from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import os
import tarfile
import torch
import xml.etree.ElementTree as ET
import numpy as np

from Object_detection_2d.data.container import Container

voc_class2id = {
    'background': 0,
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

def voc_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class VocDetectionDataset(Dataset):
    cmap = voc_cmap()
    def __init__(self, args, transform=None, target_transform=None):
        self.root = os.path.expanduser(args['data_root'])
        self.year = args['year']
        self.url = VOC_DATASET_YEAR_DICT[self.year]['url']
        self.filename = VOC_DATASET_YEAR_DICT[self.year]['filename']
        self.md5 = VOC_DATASET_YEAR_DICT[self.year]['md5']
        self.transform = transform
        self.target_transform = target_transform
        self.split = args['split']
        self.keep_difficult = args['keep_difficult']
        self.class_dict = voc_id2class
        
        base_dir = VOC_DATASET_YEAR_DICT[self.year]['base_dir']
        self.voc_root = os.path.join(self.root, base_dir)
        
        if args['download_data']:
            download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(self.voc_root):
            raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')
        
        self.img_dir = os.path.join(self.voc_root, 'JPEGImages')
        self.annotation_dir = os.path.join(self.voc_root, "Annotations")
        splits_dir = os.path.join(self.voc_root, 'ImageSets/Main')
        split_file = os.path.join(splits_dir, f'{self.split}.txt')
            
        if not os.path.exists(split_file):
            raise ValueError(
                'Wrong image_set entered! Please use split=train or split=trainval or split=val')

        with open(os.path.join(split_file), "r") as f:
            self.ids = [x.strip() for x in f.readlines()]

    def __getitem__(self, index):
        img_id = self.ids[index]
        boxes, labels, difficulties = self.parse_annotation(img_id)
        if not self.keep_difficult:
            boxes = boxes[difficulties == 0]
            labels = labels[difficulties == 0]

        img = self._read_image(img_id)
        if self.transform is not None:
            img, boxes, labels = self.transform(img, boxes, labels)
        if self.target_transform is not None:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(boxes=boxes, labels=labels)
        return img, targets, index

    def __len__(self):
        return len(self.ids)

    def get_annotation(self, index):
        img_id = self.ids[index]
        return img_id, self.parse_annotation(img_id)
    
    def parse_annotation(self, img_id):
        annotation_path = os.path.join(self.annotation_dir, f'{img_id}.xml')
        boxes = []
        labels = []
        difficulties = []
    
        objects = ET.parse(annotation_path).findall("object")
        for obj in objects:
            difficult = int(obj.find("difficult").text == "1")
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
            difficulties.append(difficult)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(difficulties, dtype=np.uint8))

    def _read_image(self, img_id):
        img_path = os.path.join(self.img_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        return np.array(img)

    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_file = os.path.join(self.annotation_dir, f'{img_id}.xml')
        annotation = ET.parse(annotation_file).getroot()
        size = annotation.find("size")
        img_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": img_info[0], "width": img_info[1]}
    
    
