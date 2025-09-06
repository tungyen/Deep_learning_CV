# SSD #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the Paskal VOC 2012 dataset. This part includes SSD (Single Stage Detector) for image object detection. For the first time use of Paskal VOC dataset, please change download in the base config to true for downloading datasets.

## Experiment ##

### Paskal VOC dataset ###
Test and evaluation are based on Paskal VOC 2012 val dataset. While training is based on 2007 + 2012 trainval dataset.

#### SSD 300 ####
| Box Weight | Box Loss | mAP |
|-----|----- |----------|
| 1.0 | SmoothL1 | 87.61% |
| 1.0 | CIoU | 72.91% |

#### SSD 512 ####
| Box Weight | Box Loss | mAP |
|-----|----- |----------|
| 1.0 | SmoothL1 | 87.23% |
| 1.0 | CIoU | 73.06% |

The command below is based on Smooth L1 loss for bounding boxes. If you want to train on IoU Loss, please change to iou_loss config.

### Training ###
```bash
torchrun --nproc_per_node=1 Object_detection_2d/SSD/pipelines/train.py --experiment ssd_300_smooth_l1 --config Object_detection_2d/SSD/config/ssd_300_regression_loss.yaml
```

### Evaluation ###
```bash
torchrun --nproc_per_node=1 Object_detection_2d/SSD/pipelines/eval.py --experiment ssd_300_smooth_l1 --config Object_detection_2d/SSD/config/ssd_300_regression_loss.yaml
```

### Testing ###
```bash
torchrun --nproc_per_node=1 Object_detection_2d/SSD/pipelines/test.py --experiment ssd_300_smooth_l1 --config Object_detection_2d/SSD/config/ssd_300_regression_loss.yaml
```

### Result of voc Dataset ###

### Paskal VOC ###

Since ciou loss leads to worsen performance, only result based on smooth-L1 loss are shown here.

#### SSD 300: ####
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Object_detection_2d/SSD/runs/ssd_300_smooth_l1/SSD_VOC.png)

#### SSD 512: ####
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Object_detection_2d/SSD/runs/ssd_512_smooth_l1/SSD_VOC.png)