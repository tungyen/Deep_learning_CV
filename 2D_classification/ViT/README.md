# ViT #
In this part, I implement three kinds of ViT based on different position encoding. Sinusoidal, Relative Distance, and Rotatory Position Encoding. First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the flower dataset.


## training ##
```bash
python train.py --dataset flower --model vit_relative
```

## testing ##
```bash
python test.py --dataset flower --model vit_rope
```


Flower image classification.

Sinusoidal Position Encoding

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/2D_classification/ViT/img/vit_sinusoidal_flower.png)

Relative Distance Position Encoding

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/2D_classification/ViT/img/vit_sinusoidal_flower.png)

Rotatory Position Encoding

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/2D_classification/ViT/img/vit_sinusoidal_flower.png)