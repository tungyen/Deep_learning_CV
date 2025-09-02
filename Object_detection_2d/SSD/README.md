# SSD #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the Paskal VOC 2012 dataset. This part includes SSD (Single Stage Detector) for image object detection. For the first time use of Paskal VOC dataset, please change download in the base config to true for downloading datasets.

## Experiment ##

### Paskal VOC dataset ###
Test and evaluation are based on Paskal VOC 2012 val dataset. While training is based on 2007 + 2012 trainval dataset.

| Box Weight | Box Loss | mAP |
|-----|----- |----------|
| 1.0 | SmoothL1 | 87.61% |
| 1.0 | IoU | % |
| 1.0 | GIoU  | % |
| 1.0 | DIoU | % |
| 1.0 | CIoU | % |

### Training ###
```bash
torchrun --nproc_per_node=1 Object_detection_2d/SSD/pipelines/train.py --experiment smooth_l1 --config Object_detection_2d/SSD/config/base.yaml
```

### Evaluation ###
```bash
torchrun --nproc_per_node=1 Object_detection_2d/SSD/pipelines/eval.py --experiment smooth_l1 --config Object_detection_2d/SSD/config/base.yaml
```

### Testing ###
```bash
torchrun --nproc_per_node=1 Object_detection_2d/SSD/pipelines/test.py --experiment smooth_l1 --config Object_detection_2d/SSD/config/base.yaml
```

### Result of voc Dataset ###

### Paskal VOC ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Object_detection_2d/SSD/runs/smooth_l1/SSD_VOC.png)