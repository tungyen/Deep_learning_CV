# ResNet/ResNeXT #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the flower dataset. In this folder, I implemented both ResNet and ResNeXt since their structure are similar.

The command is taking resnet34 as example, for other models names, please refer to utils.py
## Training ##
```bash
python train.py --dataset flower --model resnet34
python train.py --dataset cifar10 --model resnet34
python train.py --dataset cifar100 --model resnet34
```

## Evaluation ##
```bash
python eval.py --dataset flower --model resnet34
python eval.py --dataset cifar10 --model resnet34
python eval.py --dataset cifar100 --model resnet34
```

## Testing ##
```bash
python test.py --dataset flower --model resnet34
python test.py --dataset cifar10 --model resnet34
python test.py --dataset cifar100 --model resnet34
```

## ResNet34 ##

### Flower classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnet34_flower.png)

### Cifar10 classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnet34_cifar10.png)

### Cifar100 classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnet34_cifar100.png)



## ResNet50 ##

### Flower classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnet50_flower.png)

### Cifar10 classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnet50_cifar10.png)

### Cifar100 classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnet50_cifar100.png)



## ResNet101 ##

### Flower classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnet101_flower.png)

### Cifar10 classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnet101_cifar10.png)

### Cifar100 classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnet101_cifar100.png)



## ResNeXt50_32x4d ##

### Flower classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnext50_32x4d_flower.png)

### Cifar10 classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnext50_32x4d_cifar10.png)

### Cifar100 classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnext50_32x4d_cifar100.png)



## ResNeXt101_32x8d ##

### Flower classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnext101_32x8d_flower.png)

### Cifar10 classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnext101_32x8d_cifar10.png)

### Cifar100 classification ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/imgs/resnext101_32x8d_cifar100.png)