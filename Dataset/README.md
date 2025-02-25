# _Dataset for Computer Vision Tasks_ #

This is the introduction for every dataset used in this repo, and the guideline for downloading the dataset. Note that different models require different datasets. This page is only the basic guidance of downloading. You should check readme file of each model to decide which dataset is used.


# _CIFAR10_ #
Go to the [CIFAR-10 64x64](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution) for downloading, and the structure is like:
`Dataset/cifar10/cifar10-32` and `Dataset/cifar10/cifar10-64`


# _Cityscapes_ #
Go to the official website [CityScapes](https://www.cityscapes-dataset.com/downloads/) for downloading. Unzip it under `Dataset/cityscapes`.


# _LandScape_ #
Go to the [Landscape Dataset](https://www.kaggle.com/datasets/arnaud58/landscape-pictures) for downloading, and unzip it as `Dataset/landscape`


# _ModelNet40_ #
Go to the [ModelNet-40](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset) for downloading and unzip as `Dataset/ModelNet40`.


# _ShapeNet_ #
To be continue.


# _COCO_ #
Go to the [COCO official website](https://cocodataset.org/#download).  And try to use F12->`Network` tab to get the link of downloading. After downloading, unzip it under `Dataset/COCO`.
```bash
wget -c http://images.cocodataset.org/zips/train2014.zip # Train2014 for example
```

# _FlowerDataset_ #
Go to the my drive for [FlowerDataset](https://drive.google.com/file/d/1PVqNgHBQUudlIJdOcxSbq9FPTMnenDYg/view?usp=sharing). Unzip under this folder as `Dataset/flower_data`. `classIndex.json` includes the index for each flower category. And `split_data.py` is used to process the original dataset in to train/val dataset. You could directly use the pre-process train/val dataset including in the downloading folder!


# _VOCdevkit_ #
Go to [VOCdevkit](https://www.kaggle.com/datasets/wangyuhang3303/vocdevkit) for downloading. Unzip it under this folder as `Dataset/VOCdevkit`


# _PROPS-Pose-Dataset_ #
Go to [this drive link](https://drive.google.com/file/d/15rhwXhzHGKtBcxJAYMWJG7gN7BLLhyAq/view) for downloading. Unzip it under this folder as `Dataset/PROPS-Pose-Dataset`.


# _ChairDataset_ #
Go to the my drive for [ChairDataset](https://drive.google.com/file/d/12O18QBLeaeKfmXeTTNSJbNBDuCeoFoJk/view?usp=sharing). Unzip under this folder as `Dataset/Chair_dataset`.