# SegFormer #
First of all, please follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the cityscapes dataset. This part includes SegFormer model for image semantic segmentation. For the first time use of Paskal VOC dataset, please turn `download` to True in any VOC dataset config to download dataset under Dataset folder.To download different year of Paskal VOC dataset, change `year` in the config. And remember to add train_aug.txt under `Dataset/VOC/VOCdevkit/VOC2012/train_aug.txt`. The detail is also in the link above.

## Download the pretrained Mit weights ##
```bash
# Available backbone_name is {mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5}
cd Segmentation_2d
python download_pretrained_mit.py --backbone_name mit_b0
```

## Experiment ##

### Cityscapes dataset ###
| Model | CE Weight | Lovasz Weight | Boundary Weight | mIoUs |
|-------|-----|-----|-----|-----|
| SegFormer-B0 | 1.0 | None | None | 69.91% |
| SegFormer-B0 | 1.0 | 1.0 | 0.5 | 70.94% |
| SegFormer-B2 | 1.0 | None | None | 76.46% |
| SegFormer-B2 | 1.0 | 1.0 | 0.5 | % |


### Paskal VOC dataset ###
| Model | CE Weight | Lovasz Weight | Boundary Weight | mIoUs |
|-------|-----|-----|-----|-----|
| SegFormer-B0 | 1.0 | None | None | % |
| SegFormer-B0 | 1.0 | 1.0 | 0.5 | % |
| SegFormer-B2 | 1.0 | None | None | % |
| SegFormer-B2 | 1.0 | 1.0 | 0.5 | % |


You can change weight of lovasz/boundary loss in config by changing `lovasz_weight` and `boundary_weight`.
## Running the code ##
```bash
bash Segmentation_2d/run_seg_2d.sh 1 exp Segmentation_2d/config/segformer_mit_b0_cityscapes.yaml
```

## Result ##

### Result of Cityscapes Dataset ###

#### SegFormer B0 ####
CE = 1.0:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce/segformer_b0_cityscapes.png)


CE = 1.0, Lovasz = 1.0, Boundary = 0.5:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce_lovasz_bound/segformer_b0_cityscapes.png)

#### SegFormer B2 ####
CE = 1.0:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce/segformer_b2_cityscapes.png)


CE = 1.0, Lovasz = 1.0, Boundary = 0.5:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_2d/img/ce_lovasz_bound/segformer_b2_cityscapes.png)