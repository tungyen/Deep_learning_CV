# PointNet #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the chair dataset.

## Classification ##

### Training ###
```bash
python train.py --dataset modelnet40 --model pointnet_cls --class_num 40
```

### Evaluation ###
```bash
python eval.py --dataset modelnet40 --model pointnet_cls --class_num 40
```

### Testing ###
```bash
python test.py --dataset modelnet40 --model pointnet_cls --class_num 40
```

### Result of Chair Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/3D_segmentation/PointNet/img/pointnet_seg_chair.png)

## Part Segmentation ##

### Training ###
```bash
python train.py --dataset chair --model pointnet_seg --class_num 4
```

### Evaluation ###
To be continue

### Testing ###
```bash
python test.py --dataset chair --model pointnet_seg --class_num 4
```

### Result of Chair Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/3D_segmentation/PointNet/img/pointnet_seg_chair.png)
