# SSD #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the Paskal VOC 2012 dataset. This part includes SSD (Single Stage Detector) for image object detection. For the first time use of Paskal VOC dataset, please add argument --voc_download True to download dataset under Dataset folder.To download different year of Paskal VOC dataset, change voc_year in argument.

## Experiment ##

### Paskal VOC dataset ###
| Box Weight | Box Loss | mAP |
|-----|----- |----------|
| 1.0 | SmoothL1 | 54.92% |
| 1.0 | IoU | % |
| 1.0 | GIoU  | % |
| 1.0 | DIoU | % |
| 1.0 | CIoU | % |

### Training ###
```bash
torchrun --nproc_per_node=2 Object_detection_2d/SSD/train.py --experiment ckpts --config Object_detection_2d/SSD/config/base.yaml
```

### Evaluation ###
```bash
torchrun --nproc_per_node=2 Object_detection_2d/SSD/eval.py --experiment ckpts --config Object_detection_2d/SSD/config/base.yaml
```

### Testing ###
```bash
torchrun --nproc_per_node=1 Object_detection_2d/SSD/test.py --experiment ckpts --config Object_detection_2d/SSD/config/base.yaml
```

### Result of voc Dataset ###