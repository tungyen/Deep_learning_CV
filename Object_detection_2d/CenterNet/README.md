# SSD #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the Paskal VOC 2007/2012 dataset. This part includes Centernet for image object detection. For the first time use of Paskal VOC dataset, please change download in the base config to true for downloading datasets.

## Experiment ##

### Paskal VOC dataset ###
Test and evaluation are based on Paskal VOC 2012 val dataset. While training is based on 2007 + 2012 trainval dataset.

#### CenterNet 512 ####
| Width Weight | Offsets Weight | mAP |
|-----|----- |----------|
| 0.1 | 1.0 | 92.01% |

### Training ###
```bash
bash Object_detection_2d/CenterNet/run_centernet.sh centernet_voc_512 Object_detection_2d/CenterNet/config/center_net_voc.yaml
```

### Result of voc Dataset ###

### Paskal VOC ###

#### Centernet 512: ####
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Object_detection_2d/CenterNet/img/centernet_voc_512.png)