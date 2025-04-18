# Generative Adversarial Network

In this folder, I implemented `Conditional GAN`. The used dataset is shown below:
Conditional GAN: MNIST



# _Training_ #
To train the model, just use the following command for both models:

```bash
python CGAN/train.py # Change dataset for MNIST, FashionMNIST, and CIFAR
```

# _Sampling_ #
To generate reconstrcuted/synthesis images, using the following command for both models:
```bash
python CGAN/test.py # Randomly generate dataset from the latent space
```


# _Result of CGAN_ #
## _MNIST Dataset_ ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/GAN/CGAN/img/Conditional_GAN_MNIST_gen.png)


# _Result of DCGAN_ #
## _MNIST Dataset_ ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/GAN/DCGAN/img/Deep_Convolutional_GAN_MNIST_gen.png)

## _CIFAR Dataset_ ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/GAN/DCGAN/img/Deep_Convolutional_GAN_cifar_gen.png)