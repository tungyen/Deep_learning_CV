# Dataset for Computer Vision Tasks #

This is the introduction for every dataset used in this repo, and the guideline for downloading the dataset. Note that different models require different datasets. This page is only the basic guidance of downloading. You should check readme file of each model to decide which dataset is used.


## Image ##

### Cityscapes ###
Go to the official website [CityScapes](https://www.cityscapes-dataset.com/downloads/) for downloading. Unzip it under `Dataset/cityscapes`.


### LandScape ###
Go to the [Landscape Dataset](https://www.kaggle.com/datasets/arnaud58/landscape-pictures) for downloading, and unzip it as `Dataset/landscape`

### COCO ###
Go to the [COCO official website](https://cocodataset.org/#download).  And try to use F12->`Network` tab to get the link of downloading. After downloading, unzip it under `Dataset/COCO`.
```bash
wget -c http://images.cocodataset.org/zips/train2014.zip # Train2014 for example
```

### FlowerDataset ###
Go to the my drive for [FlowerDataset](https://drive.google.com/file/d/1PVqNgHBQUudlIJdOcxSbq9FPTMnenDYg/view?usp=sharing). Unzip under this folder as `Dataset/flower_data`. `classIndex.json` includes the index for each flower category. And `split_data.py` is used to process the original dataset in to train/val dataset. You could directly use the pre-process train/val dataset including in the downloading folder!


### VOCdevkit ###
Go to [VOCdevkit](https://www.kaggle.com/datasets/wangyuhang3303/vocdevkit) for downloading. Unzip it under this folder as `Dataset/VOCdevkit`. Or when running code in Segmentation_2d/Deeplabv3, you will download the VOCdevkit dataset automatically by using --voc_download True. And if you want to use Pascal VOC trainaug labels, please go to [this dropbox](https://www.dropbox.com/scl/fi/xccys1fus0utdioi7nj4d/SegmentationClassAug.zip?e=2&rlkey=0wl8iz6sc40b3qf6nidun4rez&dl=0) for downloading. And then put it under Dataset/VOC, and then run the command below to unzip to the designated path:
```bash
unzip SegmentationClassAug.zip -d VOCdevkit/VOC2012/
```
Please go to [this repo](https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/datasets/data/train_aug.txt) to download train_aug.txt file.

### Celeba ###
Go to [Celeba](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and download `img_align_celeba.zip` file. And then unzip it as Dataset/img_align_celeba


## Point Cloud ##

### ChairDataset ###
Go to the my drive for [ChairDataset](https://drive.google.com/file/d/12O18QBLeaeKfmXeTTNSJbNBDuCeoFoJk/view?usp=sharing). Unzip under this folder as `Dataset/Chair_dataset`.

### ModelNet40 ###
Go to the [ModelNet-40](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset) for downloading and unzip as `Dataset/ModelNet40`. And then run the following command:

```bash
python prepare_modelnet40.py
```

After this command, we use ModelNet40_npz as our dataset folder.


### ShapeNet ###
Go to [Kaggle](https://www.kaggle.com/datasets/mitkir/shapenet) to download ShapeNet Part Segmentation dataset. Save under Dataset/ShapeNetPart.


### S3DIS ###
To download S3DIS dataset, please follow [this link](https://github.com/open-mmlab/mmdetection3d/blob/1.0/data/s3dis/README.md). After unzip the folder, run the following command:

```bash
python prepare_s3dis.py
```

Then you might need to change the line 180389 in Annotation/Area5/hallway_6/ceiling_1 to make it 6 columns.

## 6D Pose ##

### PROPS-Pose-Dataset ###
Go to [this drive link](https://drive.google.com/file/d/15rhwXhzHGKtBcxJAYMWJG7gN7BLLhyAq/view) for downloading. Unzip it under this folder as `Dataset/PROPS-Pose-Dataset`.