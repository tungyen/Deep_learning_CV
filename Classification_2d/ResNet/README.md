# ResNet #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the flower dataset.

The command is taking resnet34 as example, for other models, please refer to utils.py
## Training ##
```bash
python train.py --dataset flower --model resnet34 --class_num 5
python train.py --dataset cifar10 --model resnet34 --class_num 10
python train.py --dataset cifar100 --model resnet34 --class_num 100
```

## Evaluation ##
```bash
python eval.py --dataset flower --model resnet34 --class_num 5
python eval.py --dataset cifar10 --model resnet34 --class_num 10
python eval.py --dataset cifar100 --model resnet34 --class_num 100
```

## Testing ##
```bash
python test.py --dataset flower --model resnet34 --class_num 5
python test.py --dataset cifar10 --model resnet34 --class_num 10
python test.py --dataset cifar100 --model resnet34 --class_num 100
```

## Flower image classification ##

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/img/resnet34_flower.png)

## Cifar10 classification ##

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/img/resnet34_cifar10.png)

## Flower image classification ##

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/ResNet/img/resnet34_cifar100.png)