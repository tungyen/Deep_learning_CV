# PointNet/PointNet++ #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the chair dataset, ModelNet40, S3DIS, ShapeNet, and ScanNet.

For PointNet++, you need to compile the CUDA code:
```bash
cd model/utils
python setup.py build_ext --inplace
```

## Classification ##

### Training ###
```bash
python train.py --dataset modelnet40 --model pointnet_cls
```

### Evaluation ###
```bash
python eval.py --dataset modelnet40 --model pointnet_cls
```

### Testing ###
```bash
python test.py --dataset modelnet40 --model pointnet_cls --batch_size 4
```

### Result of Modelnet40 Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_cls_modelnet40.png)

## Part Segmentation ##

### Training ###
```bash
python train.py --dataset chair --model pointnet_seg
```

### Evaluation ###
```bash
python eval.py --dataset chair --model pointnet_seg
```

### Testing ###
```bash
python test.py --dataset chair --model pointnet_seg --batch_size 6
```

### Result of Chair Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Segmentation_3d/PointNet/imgs/pointnet_seg_chair.png)
