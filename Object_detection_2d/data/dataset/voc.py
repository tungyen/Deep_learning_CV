from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import os
import tarfile
import torch
import xml.etree.ElementTree as ET
import numpy as np
import bisect

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
    def __init__(self, args, stage="train", transform=None, target_transform=None):
        self.root = os.path.expanduser(args['data_root'])
        self.years = args[stage]['years']
        self.urls = [VOC_DATASET_YEAR_DICT[year]['url'] for year in self.years]
        self.filenames = [VOC_DATASET_YEAR_DICT[year]['filename'] for year in self.years]
        self.md5 = [VOC_DATASET_YEAR_DICT[year]['md5'] for year in self.years]
        self.transform = transform
        self.target_transform = target_transform
        self.splits = args[stage]['splits']
        self.keep_difficult = args['keep_difficult']
        self.class_dict = voc_id2class
        
        base_dirs = [VOC_DATASET_YEAR_DICT[year]['base_dir'] for year in self.years]
        self.voc_roots = [os.path.join(self.root, base_dir) for base_dir in base_dirs]
        
        for i, download in enumerate(args[stage]['download_data']):
            if download:
                download_extract(self.urls[i], self.root, self.filenames[i], self.md5[i])
        for voc_root in self.voc_roots:
            if not os.path.isdir(voc_root):
                raise RuntimeError('Dataset not found or corrupted.' + ' You can use download=True to download it')
        
        self.img_dirs = [os.path.join(voc_root, 'JPEGImages') for voc_root in self.voc_roots]
        self.annotation_dirs = [os.path.join(voc_root, "Annotations") for voc_root in self.voc_roots]
        splits_dirs = [os.path.join(voc_root, 'ImageSets/Main') for voc_root in self.voc_roots]
        split_files = [os.path.join(splits_dir, f'{split}.txt') for split, splits_dir in zip(self.splits, splits_dirs)]

        for split_file in split_files: 
            if not os.path.exists(split_file):
                raise ValueError(
                    'Wrong image_set entered! Please use split=train or split=trainval or split=val')
        self.ids = []
        size = 0
        self.sizes = []
        for split_file in split_files: 
            with open(os.path.join(split_file), "r") as f:
                id_cur = [x.strip() for x in f.readlines()]
                size += len(id_cur)
                self.sizes.append(size)
                self.ids.extend(id_cur)

    def __getitem__(self, index):
        img_id = self.ids[index]
        boxes, labels, difficulties = self.parse_annotation(img_id, index)
        if not self.keep_difficult:
            boxes = boxes[difficulties == 0]
            labels = labels[difficulties == 0]

        img = self._read_image(img_id, index)
        if self.transform is not None:
            img, boxes, labels = self.transform(img, boxes, labels)
        result = {
            "boxes": boxes,
            "labels": labels
        }
        if self.target_transform is not None:
            result = self.target_transform(boxes, labels)
        targets = Container(result)
        return img, targets, index

    def __len__(self):
        return len(self.ids)

    def get_annotation(self, index):
        img_id = self.ids[index]
        return img_id, self.parse_annotation(img_id, index)
    
    def parse_annotation(self, img_id, index):
        annotation_dir = self.annotation_dirs[bisect.bisect_right(self.sizes, index)]
        annotation_path = os.path.join(annotation_dir, f'{img_id}.xml')
        # print("Img: ", img_id)
        # print("Index: ", index)
        # print("Anno Path: ", annotation_path)
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

    def _read_image(self, img_id, index):
        img_dir = self.img_dirs[bisect.bisect_right(self.sizes, index)]
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        return np.array(img)

    def get_img_info(self, index):
        img_id = self.ids[index]
        annotation_dir = self.annotation_dirs[bisect.bisect_right(self.sizes, index)]
        annotation_file = os.path.join(annotation_dir, f'{img_id}.xml')
        annotation = ET.parse(annotation_file).getroot()
        size = annotation.find("size")
        img_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": img_info[0], "width": img_info[1]}
    
    
