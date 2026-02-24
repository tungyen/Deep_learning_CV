# SegFormer #
First of all, please follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the cityscapes dataset. This part includes SegFormer model for image semantic segmentation. For the first time use of Paskal VOC dataset, please turn `download` to True in any VOC dataset config to download dataset under Dataset folder.To download different year of Paskal VOC dataset, change `year` in the config. And remember to add train_aug.txt under `Dataset/VOC/VOCdevkit/VOC2012/train_aug.txt`. The detail is also in the link above.

## Experiment ##

### Paskal VOC dataset ###
| Model | Dataset | mIoUs |
|-------|-----|-----|
| SegFormer | VOC 512 | % |


You can change weight of lovasz/boundary loss in config by changing `lovasz_weight` and `boundary_weight`.
## Running the code ##
```bash
bash Segmentation_2d/run_seg_2d.sh 1 exp Segmentation_2d/config/segformer_voc.yaml
```

## Result ##