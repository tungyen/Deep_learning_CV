# ViT #
In this part, I implement three kinds of ViT based on different position encoding. Sinusoidal, Relative Distance, and Rotatory Position Encoding. First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the flower dataset. And I also train the model based on cifar10 and cifar100 dataset.

Here is the pipeline command for ViT with relative distance position embedding, please change to other type of position embedding on your will.

```bash
bash Classification_2d/run_cls_model.sh 1 test Classification_2d/config/vit_flower.yaml
```


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