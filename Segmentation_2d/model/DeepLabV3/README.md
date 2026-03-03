# DeepLabV3 #
First of all, please follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the cityscapes dataset. This part includes DeepLabV3 and DeepLabV3+ model for image semantic segmentation. For the first time use of Paskal VOC dataset, please turn `download` to True in any VOC dataset config to download dataset under Dataset folder.To download different year of Paskal VOC dataset, change `year` in the config. And remember to add train_aug.txt under `Dataset/VOC/VOCdevkit/VOC2012/train_aug.txt`. The detail is also in the link above.

## Experiment ##

### CityScapes dataset ###
| Model | CE Weight | Lovasz Weight | Boundary Weight | mIoUs |
|-------|-----|----------|-------------------|----------|
| DeepLabV3 | 1.0 | None | None | 64.65% |
| DeepLabV3 | 1.0 | 1.0 | None | 66.41% |
| DeepLabV3 | 1.0 | 1.0 | 0.5 | 67.69% |
| DeepLabV3++ | 1.0 | None | None | 69.91% |
| DeepLabV3++ | 1.0 | 1.5 | None | 70.63% |
| DeepLabV3++ | 1.0 | 1.5 | 0.5 | 71.41% |


### Paskal VOC dataset ###
| Model | CE Weight | Lovasz Weight | Boundary Weight | Backbone | mIoUs |
|-------|-----|----------|-------------------|----------|-------|
| DeepLabV3 | 1.0 | None | None | ResNet-101 | 70.19% |
| DeepLabV3 | 1.0 | 1.0 | None | ResNet-101 | 69.19% |
| DeepLabV3 | 1.0 | 1.0 | 0.5 | ResNet-101 | 69.37% |
| DeepLabV3 | 1.0 | None | None | ResNet-152 | 72.83% |
| DeepLabV3 | 1.0 | 1.0 | None | ResNet-152 | 72.41% |
| DeepLabV3 | 1.0 | 1.0 | 0.5 | ResNet-152 | 71.94% |
| DeepLabV3++ | 1.0 | None | None | ResNet-101 | 73.31% |
| DeepLabV3++ | 1.0 | 1.0 | None | ResNet-101 | 73.65% |
| DeepLabV3++ | 1.0 | 1.0 | 0.5 | ResNet-101 | 69.93% |
| DeepLabV3++ | 1.0 | None | None | ResNet-152 | 77.09% |
| DeepLabV3++ | 1.0 | 1.0 | None | ResNet-152 | 76.31% |
| DeepLabV3++ | 1.0 | 1.0 | 0.5 | ResNet-152 | 76.5% |

You can change weight of lovasz/boundary loss in config by changing `lovasz_weight` and `boundary_weight`.
## Running the code ##
```bash
# DeepLabV3
bash Segmentation_2d/run_seg_2d.sh 1 exp Segmentation_2d/config/deeplabv3_cityscapes.yaml

# DeepLabV3+
bash Segmentation_2d/run_seg_2d.sh 1 exp Segmentation_2d/config/deeplabv3plus_cityscapes.yaml
```

## Result ##

### Result of Cityscapes Dataset ###

#### DeepLabV3 ####
CE = 1.0:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce/deeplabv3_cityscapes.png)

CE = 1.0, Lovasz = 1.5, Boundary = 0.5:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce_lovasz_bound_default/deeplabv3_cityscapes.png)

#### DeepLabV3+ ####
CE = 1.0:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce/deeplabv3plus_cityscapes.png)

CE = 1.0, Lovasz = 1.5, Boundary = 0.5:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce_lovasz_bound_default/deeplabv3plus_cityscapes.png)

### Result of Paskal VOC 2012 Dataset ###

#### DeepLabV3 ####
CE = 1.0:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce/deeplabv3_voc_2012_aug.png)

CE = 1.0, Lovasz = 1.5, Boundary = 0.5:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce_lovasz_bound_default/deeplabv3_voc_2012_aug.png)

#### DeepLabV3+ ####
CE = 1.0:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce/deeplabv3plus_voc_2012_aug.png)

CE = 1.0, Lovasz = 1.5, Boundary = 0.5:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce_lovasz_bound_default/deeplabv3plus_voc_2012_aug.png)