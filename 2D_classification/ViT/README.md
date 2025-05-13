# ViT #
In this part, I implement three kinds of ViT based on different position encoding. Sinusoidal, Relative Distance, and Rotatory Position Encoding. First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the flower dataset. And I also train the model based on cifar10 and cifar100 dataset.

Here is the pipeline command for ViT with relative distance position embedding, please change to other type of position embedding on your will.
## training ##
```bash
python train.py --dataset flower --model vit_relative --patch_size 16 --img_size 224 --class_num 5
python train.py --dataset cifar10 --model vit_relative --patch_size 4 --img_size 32 --class_num 10
python train.py --dataset ciar100 --model vit_relative --patch_size 4 --img_size 32 --class_num 100
```

## evaluating ##
```bash
python eval.py --dataset flower --model vit_relative --patch_size 16 --img_size 224 --class_num 5
python eval.py --dataset cifar10 --model vit_relative --patch_size 4 --img_size 32 --class_num 10
python eval.py --dataset ciar100 --model vit_relative --patch_size 4 --img_size 32 --class_num 100
```

## testing ##
```bash
python test.py --dataset flower --model vit_relative --patch_size 16 --img_size 224 --class_num 5
python test.py --dataset cifar10 --model vit_relative --patch_size 4 --img_size 32 --class_num 10
python test.py --dataset ciar100 --model vit_relative --patch_size 4 --img_size 32 --class_num 100
```


## Flower image classification ##

### _Sinusoidal Position Encoding_ ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/2D_classification/ViT/img/vit_sinusoidal_flower.png)

### _Relative Distance Position Encoding_ ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/2D_classification/ViT/img/vit_relative_flower.png)

### _Rotatory Position Encoding_ ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/2D_classification/ViT/img/vit_rope_flower.png)


## CIFAR10 classification ##

### _Sinusoidal Position Encoding_ ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/2D_classification/ViT/img/vit_sinusoidal_cifar10.png)

### _Relative Distance Position Encoding_ ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/2D_classification/ViT/img/vit_relative_cifar10.png)

### _Rotatory Position Encoding_ ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/2D_classification/ViT/img/vit_rope_cifar10.png)