# SSD #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the Paskal VOC 2007/2012 dataset. This part includes SSD (Single Stage Detector) for image object detection. For the first time use of Paskal VOC dataset, please change download in the base config to true for downloading datasets.

## Experiment ##

### Paskal VOC dataset ###
Test and evaluation are based on Paskal VOC 2012 val dataset. While training is based on 2007 + 2012 trainval dataset.

#### SSD 300 ####
| Box Weight | Box Loss | mAP |
|-----|----- |----------|
| 1.0 | SmoothL1 | 94.77% |
| 1.0 | CIoU | 90.41% |

#### SSD 512 ####
| Box Weight | Box Loss | mAP |
|-----|----- |----------|
| 1.0 | SmoothL1 | 94.9% |
| 1.0 | CIoU | 90.55% |

The command below is based on Smooth L1 loss for bounding boxes. If you want to train on IoU Loss, please change to iou_loss config.

### Training ###
```bash
bash Object_detection_2d/SSD/run_ssd.sh ssd_300_smooth_l1 Object_detection_2d/SSD/config/ssd_300_regression_loss.yaml
```

### Result of voc Dataset ###

### Paskal VOC ###

Since ciou loss leads to worsen performance, only result based on smooth-L1 loss are shown here.

#### SSD 300: ####
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Object_detection_2d/SSD/img/SSD_VOC_300_regression.png)

#### SSD 512: ####
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Object_detection_2d/SSD/img/SSD_VOC_512_regression.png)