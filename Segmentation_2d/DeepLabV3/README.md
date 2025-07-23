# DeepLabV3 #
First of all, please follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the cityscapes dataset. This part includes DeepLabV3 and DeepLabV3+ model for image semantic segmentation. For the first time use of Paskal VOC dataset, please add argument --voc_download True to download dataset under Dataset folder.To download different year of Paskal VOC dataset, change voc_year in argument. And remember to add train_aug.txt under Dataset/VOC/VOCdevkit/VOC2012/train_aug.txt. The detail is also in the link above.

## Experiment ##

### CityScapes dataset ###
| Model  | CE Weight | Lovasz Weight | Boundary Weight | mIoUs |
|-------|-----|----------|-------------------|----------|
| DeepLabV3 | 1.0  | None | None   | 63.43%   |
| DeepLabV3   | 1.0  | 0.5   | None      | 64.92% |
| DeepLabV3 | 1.0  | 1.0    | None    | 65.38%   |
| DeepLabV3 | 1.0  | 1.0    | 0.3    | 65.49%   |
| DeepLabV3 | 1.0  | 1.5   | 0.5    | 66.55%   |

## Training ##
```bash
# DeepLabV3
python Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset cityspaces --model deeplabv3 --backbone resnet101 
python Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3 --backbone resnet101
python Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3 --backbone resnet101

# DeepLabV3+
python Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset cityspaces --model deeplabv3plus --backbone resnet101 
python Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3plus --backbone resnet101
python Segmentation_2d/DeepLabV3/train.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3plus --backbone resnet101
```

## Evaluation ##
```bash
# DeepLabV3
python Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset cityspaces --model deeplabv3 --backbone resnet101 
python Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3 --backbone resnet101
python Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3 --backbone resnet101

# DeepLabV3+
python Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset cityspaces --model deeplabv3plus --backbone resnet101 
python Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3plus --backbone resnet101
python Segmentation_2d/DeepLabV3/eval.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3plus --backbone resnet101
```

## Test ##
```bash
# DeepLabV3
python Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset cityspaces --model deeplabv3 --backbone resnet101 
python Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3 --backbone resnet101
python Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3 --backbone resnet101

# DeepLabV3+
python Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset cityspaces --model deeplabv3plus --backbone resnet101 
python Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset voc --voc_year 2012 --model deeplabv3plus --backbone resnet101
python Segmentation_2d/DeepLabV3/test.py --experiment ckpts --dataset voc --voc_year 2012_aug --model deeplabv3plus --backbone resnet101
```

## Result ##

### Result of Cityscapes Dataset ###

#### DeepLabV3 ####
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/DeepLabV3/imgs/deeplabv3_cityscapes.png)

#### DeepLabV3+ ####
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/DeepLabV3/imgs/deeplabv3plus_cityscapes.png)

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