# PointNet/PointNet++ #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the chair dataset, ModelNet40, S3DIS, ShapeNet, and ScanNet.

For ShapeNet, there are two default arguments, class_choice and normal_channel. The first one control the category included in the dataset, while normal vector feature will be used when normal_channel is set as True.

## Setup ##
For PointNet++, you need to compile the CUDA code:
```bash
cd model/utils
python setup.py build_ext --inplace
```

## Classification ##

For classification task, ShapeNet and ModelNet40 are used. The following command is based on ModelNet40.
### Training ###
```bash
python train.py --dataset modelnet40 --model pointnet_cls
python train.py --dataset modelnet40 --model pointnet_plus_ssg_cls
python train.py --dataset modelnet40 --model pointnet_plus_msg_cls
```

### Evaluation ###
```bash
python eval.py --dataset modelnet40 --model pointnet_cls
python eval.py --dataset modelnet40 --model pointnet_plus_ssg_cls
python eval.py --dataset modelnet40 --model pointnet_plus_msg_cls
```

### Testing ###
```bash
python test.py --dataset modelnet40 --model pointnet_cls
python test.py --dataset modelnet40 --model pointnet_plus_ssg_cls
python test.py --dataset modelnet40 --model pointnet_plus_msg_cls
```

### Result of Modelnet40 Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_cls_modelnet40.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_ssg_cls_modelnet40.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_msg_cls_modelnet40.png)

### Result of ShapeNet Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_cls_shapenet.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_ssg_cls_shapenet.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_msg_cls_shapenet.png)

## Part Segmentation ##

In this part, Chair and ShapeNetPart are used. The following command is based on Chair Dataset.
### Training ###
```bash
python train.py --dataset chair --model pointnet_seg
python train.py --dataset chair --model pointnet_plus_ssg_seg
python train.py --dataset chair --model pointnet_plus_msg_seg
```

### Evaluation ###
```bash
python eval.py --dataset chair --model pointnet_seg
python eval.py --dataset chair --model pointnet_plus_ssg_seg
python eval.py --dataset chair --model pointnet_plus_msg_seg
```

### Testing ###
```bash
python test.py --dataset chair --model pointnet_seg
python test.py --dataset chair --model pointnet_plus_ssg_seg
python test.py --dataset chair --model pointnet_plus_msg_seg
```

### Result of Chair Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_seg_chair.png)
