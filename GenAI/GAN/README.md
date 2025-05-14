# Generative Adversarial Network #

In this folder, I implemented `Conditional GAN`. The used dataset is shown below:
Conditional GAN: MNIST



## Training ##
To train the model, just use the following command for both models:

```bash
python CGAN/train.py # Change dataset for MNIST, FashionMNIST, and CIFAR
```

## Sampling ##
To generate reconstrcuted/synthesis images, using the following command for both models:
```bash
python CGAN/test.py # Randomly generate dataset from the latent space
```


## Result of CGAN ##
### MNIST Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/GAN/CGAN/img/Conditional_GAN_MNIST_gen.png)


## Result of DCGAN ##
### MNIST Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/GAN/DCGAN/img/Deep_Convolutional_GAN_MNIST_gen.png)

### CIFAR Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/GAN/DCGAN/img/Deep_Convolutional_GAN_cifar_gen.png)