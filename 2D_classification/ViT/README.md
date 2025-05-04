# ViT #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the flower dataset.


## training ##
```bash
python train.py --dataset flower
```

## testing ##
```bash
python test.py --dataset flower
```

Now the code only predict the result based on 1 single image. The code for predict a batch of image will be released soon.


Here is the simple test result of ViT for flower image classification.

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/2D_classification/ViT/img/vit_flower.png)