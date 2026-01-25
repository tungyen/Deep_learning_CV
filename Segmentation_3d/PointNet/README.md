# PointNet/PointNet++ #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the chair dataset, ModelNet40, S3DIS, and ShapeNet.

For ShapeNet, there are two default arguments, class_choice and normal_channel. The first one control the category included in the dataset, while normal vector feature will be used when normal_channel is set as True.

## Setup ##
For PointNet++, you need to compile the CUDA code:
```bash
cd model/utils
python setup.py build_ext --inplace
```

## Experiment ##

### Chair ###
| Model | CE Weight | Lovasz Weight | mIoUs |
|-------|-----|--------|----------|
| PointNet | 1.0 | 1.5 | 80.75% |
| PointNet++ SSG | 1.0 | 1.5 | 83.75% |
| PointNet++ MSG | 1.0 | 1.5 | 88.58% |

### ModelNet40 ###
| Model | CE Weight| Lovasz Weight | Precision | Recall
|-------|-----|--------|------|---------|
| PointNet | 1.0 | 1.5 | 93.68% | 91.74% |
| PointNet++ SSG | 1.0 | 1.5 | 94.53% | 92.84% |
| PointNet++ MSG | 1.0 | 1.5 | 95.52% | 94.01% |

### ShapeNetPart ###
| Model | CE Weight| Lovasz Weight | Instance mIoUs | Class mIoUs
|-------|-----|--------|----------|--------|
| PointNet | 1.0 | 1.5 | 68.28% | 59.88%
| PointNet++ SSG | 1.0 | 1.5 | 68.24% | 63.31% |
| PointNet++ MSG | 1.0 | 1.5 | 70.33% | 59.62% |

### S3DIS ###
| Model | CE Weight | Lovasz Weight | mIoUs |
|-------|-----|--------|----------|
| PointNet | 1.0 | 1.5 | 92.44% |
| PointNet++ SSG | 1.0 | 1.5 | 92.67% |
| PointNet++ MSG | 1.0 | 1.5 | 94.73% |

## Classification ##

For classification task, ModelNet40 is used.
### Training ###
```bash
# PointNet
bash Segmentation_3d/PointNet/run_pointnet.sh 1 test2 Segmentation_3d/config/pointnet_modelnet40.yaml 

# PointNetPlusSsg
bash Segmentation_3d/PointNet/run_pointnet.sh 1 test2 Segmentation_3d/config/pointnet_plus_ssg_modelnet40.yaml 

# PointNetPlusMsg
bash Segmentation_3d/PointNet/run_pointnet.sh 1 test2 Segmentation_3d/config/pointnet_plus_msg_modelnet40.yaml 
```

### Result of Modelnet40 Dataset ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_modelnet40_cls.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_ssg_modelnet40_cls.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_msg_modelnet40_cls.png)

## Part Segmentation ##

In this part, ShapeNetPart is used.
### Training ###
```bash
# PointNet
bash Segmentation_3d/PointNet/run_pointnet.sh 1 test2 Segmentation_3d/config/pointnet_shapenet.yaml 

# PointNetPlusSsg
bash Segmentation_3d/PointNet/run_pointnet.sh 1 test2 Segmentation_3d/config/pointnet_plus_ssg_shapenet.yaml 

# PointNetPlusMsg
bash Segmentation_3d/PointNet/run_pointnet.sh 1 test2 Segmentation_3d/config/pointnet_plus_msg_shapenet.yaml 
```

### Result of Shapenet Dataset ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_shapenet_partseg.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_ssg_shapenet_partseg.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_msg_shapenet_partseg.png)

## Semantic Segmentation ##

In this part, Chair Dataset and S3DIS are used. Following command is based on Chair Dataset. In semantic segmentation task, we use Lovasz Softmax Loss and Focal Loss:
### Training ###
```bash
# PointNet
bash Segmentation_3d/PointNet/run_pointnet.sh 1 test2 Segmentation_3d/config/pointnet_chair.yaml 

# PointNetPlusSsg
bash Segmentation_3d/PointNet/run_pointnet.sh 1 test2 Segmentation_3d/config/pointnet_plus_ssg_chair.yaml 

# PointNetPlusMsg
bash Segmentation_3d/PointNet/run_pointnet.sh 1 test2 Segmentation_3d/config/pointnet_plus_msg_chair.yaml 
```

### Result of Chair Dataset ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_chair_semseg.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_ssg_chair_semseg.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_plus_msg_chair_semseg.png)