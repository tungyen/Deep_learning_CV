# PointNet/PointNet++ #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the chair dataset, ModelNet40, S3DIS, and ShapeNet.

For ShapeNet, there are two default arguments, class_choice and normal_channel. The first one control the category included in the dataset, while normal vector feature will be used when normal_channel is set as True.

## Setup ##
For PointNet++, you need to compile the CUDA code:
```bash
cd model/utils
python setup.py build_ext --inplace
```

## Classification ##

For classification task, ModelNet40 is used.
### Training ###
```bash
python Segmentation_3d/PointNet/train.py --experiment ckpts --dataset modelnet40 --model pointnet
python Segmentation_3d/PointNet/train.py --experiment ckpts --dataset modelnet40 --model pointnet_plus_ssg
python Segmentation_3d/PointNet/train.py --experiment ckpts --dataset modelnet40 --model pointnet_plus_msg
```

### Evaluation ###
```bash
python Segmentation_3d/PointNet/eval.py --experiment ckpts --dataset modelnet40 --model pointnet
python Segmentation_3d/PointNet/eval.py --experiment ckpts --dataset modelnet40 --model pointnet_plus_ssg
python Segmentation_3d/PointNet/eval.py --experiment ckpts --dataset modelnet40 --model pointnet_plus_msg
```

### Testing ###
```bash
python Segmentation_3d/PointNet/test.py --experiment ckpts --dataset modelnet40 --model pointnet
python Segmentation_3d/PointNet/test.py --experiment ckpts --dataset modelnet40 --model pointnet_plus_ssg
python Segmentation_3d/PointNet/test.py --experiment ckpts --dataset modelnet40 --model pointnet_plus_msg
```

### Result of Modelnet40 Dataset ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_modelnet40_cls.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_ssg_modelnet40_cls.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_msg_modelnet40_cls.png)

## Part Segmentation ##

In this part, ShapeNetPart is used.
### Training ###
```bash
python Segmentation_3d/PointNet/train.py --experiment ckpts --dataset shapenet --model pointnet
python Segmentation_3d/PointNet/train.py --experiment ckpts --dataset shapenet --model pointnet_plus_ssg
python Segmentation_3d/PointNet/train.py --experiment ckpts --dataset shapenet --model pointnet_plus_msg
```

### Evaluation ###
```bash
python Segmentation_3d/PointNet/eval.py --experiment ckpts --dataset shapenet --model pointnet
python Segmentation_3d/PointNet/eval.py --experiment ckpts --dataset shapenet --model pointnet_plus_ssg
python Segmentation_3d/PointNet/eval.py --experiment ckpts --dataset shapenet --model pointnet_plus_msg
```

### Testing ###
```bash
python Segmentation_3d/PointNet/test.py --experiment ckpts --dataset shapenet --model pointnet
python Segmentation_3d/PointNet/test.py --experiment ckpts --dataset shapenet --model pointnet_plus_ssg
python Segmentation_3d/PointNet/test.py --experiment ckpts --dataset shapenet --model pointnet_plus_msg
```

### Result of Shapenet Dataset ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_shapenet_partseg.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_ssg_shapenet_partseg.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_msg_shapenet_partseg.png)

## Semantic Segmentation ##

In this part, Chair Dataset and S3DIS are used. Following command is based on Chair Dataset. In semantic segmentation task, we use Lovasz Softmax Loss and Focal Loss:
### Training ###
```bash
python Segmentation_3d/PointNet/train.py --experiment ckpts --dataset chair --model pointnet --loss_func focal_lovasz
python Segmentation_3d/PointNet/train.py --experiment ckpts --dataset chair --model pointnet_plus_ssg --loss_func focal_lovasz
python Segmentation_3d/PointNet/train.py --experiment ckpts --dataset chair --model pointnet_plus_msg --loss_func focal_lovasz
```

### Evaluation ###
```bash
python Segmentation_3d/PointNet/eval.py --experiment ckpts --dataset chair --model pointnet
python Segmentation_3d/PointNet/eval.py --experiment ckpts --dataset chair --model pointnet_plus_ssg
python Segmentation_3d/PointNet/eval.py --experiment ckpts --dataset chair --model pointnet_plus_msg
```

### Testing ###
```bash
python Segmentation_3d/PointNet/test.py --experiment ckpts --dataset chair --model pointnet
python Segmentation_3d/PointNet/test.py --experiment ckpts --dataset chair --model pointnet_plus_ssg
python Segmentation_3d/PointNet/test.py --experiment ckpts --dataset chair --model pointnet_plus_msg
```

### Result of Chair Dataset ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_chair_semseg.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_ssg_chair_semseg.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_msg_chair_semseg.png)


### Result of S3DIS ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_s3dis_semseg.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_ssg_s3dis_semseg.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_msg_s3dis_semseg.png)