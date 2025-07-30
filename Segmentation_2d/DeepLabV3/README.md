# DeepLabV3 #
First of all, please follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the cityscapes dataset. This part includes DeepLabV3 and DeepLabV3+ model for image semantic segmentation. For the first time use of Paskal VOC dataset, please add argument --voc_download True to download dataset under Dataset folder.To download different year of Paskal VOC dataset, change voc_year in argument. And remember to add train_aug.txt under Dataset/VOC/VOCdevkit/VOC2012/train_aug.txt. The detail is also in the link above.

## Experiment ##

### CityScapes dataset ###
| Model  | CE Weight | Lovasz Weight | Boundary Weight | mIoUs |
|-------|-----|----------|-------------------|----------|
| DeepLabV3 | 1.0  | None | None   | 64.65%   |
| DeepLabV3   | 1.0  | 1.5   | None      | 66.41% |
| DeepLabV3 | 1.0  | 1.5   | 0.5    | 67.69%   |
| DeepLabV3++ | 1.0  | None | None   | 69.91%   |
| DeepLabV3++ | 1.0  | 1.5    | None    | 70.63%   |
| DeepLabV3++ | 1.0  | 1.5   | 0.5    | 71.41%   |


### Paskal VOC dataset ###
| Model  | CE Weight | Lovasz Weight | Boundary Weight | mIoUs |
|-------|-----|----------|-------------------|----------|
| DeepLabV3 | 1.0  | None | None   | 77.55%   |
| DeepLabV3   | 1.0  | 1.5   | None      | 78.85% |
| DeepLabV3 | 1.0  | 1.5   | 0.5    | 79.05%   |
| DeepLabV3++ | 1.0  | None | None   | %   |
| DeepLabV3++ | 1.0  | 1.5    | None    | %   |
| DeepLabV3++ | 1.0  | 1.5   | 0.5    | %   |

Note that the following command is based on 2 gpu training. You can change weight of lovasz/boundary loss in command by specifying lovasz_weight and boundary_weight.
## Training ##
```bash
# DeepLabV3
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset cityspaces --model deeplabv3 --backbone resnet101 
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3 --backbone resnet101
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3 --backbone resnet101

# DeepLabV3+
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset cityspaces --model deeplabv3plus --backbone resnet101 
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3plus --backbone resnet101
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3plus --backbone resnet101
```

## Evaluation ##
```bash
# DeepLabV3
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset cityspaces --model deeplabv3 --backbone resnet101 
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3 --backbone resnet101
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3 --backbone resnet101

# DeepLabV3+
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset cityspaces --model deeplabv3plus --backbone resnet101 
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3plus --backbone resnet101
torchrun --nproc_per_node=2 Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3plus --backbone resnet101
```

## Test ##
```bash
# DeepLabV3
torchrun --nproc_per_node=1 Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset cityspaces --model deeplabv3 --backbone resnet101 
torchrun --nproc_per_node=1 Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3 --backbone resnet101
torchrun --nproc_per_node=1 Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3 --backbone resnet101

# DeepLabV3+
torchrun --nproc_per_node=1 Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset cityspaces --model deeplabv3plus --backbone resnet101 
torchrun --nproc_per_node=1 Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3plus --backbone resnet101
torchrun --nproc_per_node=1 Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3plus --backbone resnet101
```

## Result ##

### Result of Cityscapes Dataset ###

#### DeepLabV3 ####
CE = 1.0:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/DeepLabV3/ce/deeplabv3_cityscapes.png)

CE = 1.0, Lovasz = 1.5, Boundary = 0.5:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/DeepLabV3/ce_lovasz_bound_default/deeplabv3_cityscapes.png)

#### DeepLabV3+ ####
CE = 1.0:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/DeepLabV3/ce/deeplabv3plus_cityscapes.png)

CE = 1.0, Lovasz = 1.5, Boundary = 0.5:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/DeepLabV3/ce_lovasz_bound_default/deeplabv3plus_cityscapes.png)

### Result of Paskal VOC 2012 Dataset ###

#### DeepLabV3 ####
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/DeepLabV3/imgs/deeplabv3_voc_2012.png)

#### DeepLabV3+ ####
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/DeepLabV3/imgs/deeplabv3plus_voc_2012.png)

### Result of Paskal VOC 2012_aug Dataset ###

#### DeepLabV3 ####
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/DeepLabV3/imgs/deeplabv3_voc_2012_aug.png)

#### DeepLabV3+ ####
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/DeepLabV3/imgs/deeplabv3plus_voc_2012_aug.png)