# ViT & Swin Transformer #
In this part, I implement three kinds of `ViT` based on different position encoding and `Swin Transformer`. `Sinusoidal`, `Relative Distance`, and `Rotatory Position Encoding`. First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the flower dataset. And I also train the model based on cifar10 and cifar100 dataset.

Here is the pipeline command for ViT, please change to other config on your will.

```bash
bash Classification_2d/run_cls_model.sh 1 test Classification_2d/config/vit_sine_flower.yaml
```
## Experiment (Only on the flower dataset) ##

### Flower dataset ###

| Position Encoding | Precision | Recall | F1 |
|-----|----- |----------|
| Sinusoidal | 68.09% | 67.68% | 67.41% |
| Relative Distance| 69.49% | 69.37% | 69.02% |
| Rotatory | 69.34% | 69.03% | 68.93% |
| Swin Transformer | 64.66% | 63.45% | 63.62% |


## Flower image classification ##

### Sinusoidal Position Encoding ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/imgs/vit_sinusoidal_flower.png)

### Relative Distance Position Encoding ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/imgs/vit_relative_flower.png)

### Rotatory Position Encoding ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/imgs/vit_rope_flower.png)


## CIFAR10 classification ##

### Sinusoidal Position Encoding ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/imgs/vit_sinusoidal_cifar10.png)

### Relative Distance Position Encoding ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/imgs/vit_relative_cifar10.png)

### Rotatory Position Encoding ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/imgs/vit_rope_cifar10.png)


## CIFAR100 classification ##

### Sinusoidal Position Encoding ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/imgs/vit_sinusoidal_cifar100.png)

### Relative Distance Position Encoding ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/imgs/vit_relative_cifar100.png)

### Rotatory Position Encoding ###

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/Classification_2d/imgs/vit_rope_cifar100.png)