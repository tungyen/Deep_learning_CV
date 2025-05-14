# PixelCNN #

In this folder, I implemented `PixelCNN` and `Gated Pixel CNN`. All models could be trained on the MNIST dataset and CIFAR dataset (my code include implementation for RGB channels). For Gated PixelCNN, I also include conditional control for image generation. Note that this folder is just a simple implementation for VQVAE generation task so that the dataset is quite simple. Future work includes experiments on bigger dataset


## Training ##
To train the model, just use the following command for both models:

```bash
python PixelCNN/train.py
python Gated_PixelCNN/train.py
```

## Sampling ##
To generate synthesis images, using the following command for both models:
```bash
python PixelCNN/test.py
python Gated_PixelCNN/test.py
```


## Result of PixelCNN ##
With color level 4:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/PixelCNN/PixelCNN/img/pixelCnn_MNIST_4.png)

With color level 256:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/PixelCNN/PixelCNN/img/pixelCnn_MNIST_256.png)

## Result of Gate PixelCNN ##
With color level 4:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/PixelCNN/Gated_PixelCNN/img/gated_pixelCnn_MNIST_4.png)

With color level 256:

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/PixelCNN/Gated_PixelCNN/img/gated_pixelCnn_MNIST_256.png)
