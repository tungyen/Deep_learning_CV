# Variational Autoencoder #

In this folder, I implemented `Variational Autoencoder` and `Vector Quantization Variational Autoencoder`. And `Vector Quantization Variational Autoencoder` applies `Gated PixelCNN` as the prior generation model for sampling. All models could be trained on the [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), MNIST, Fashion MNIST, and CIFAR dataset.

Note that VAE_pi is a project from an company online assessment. The goal is specifically for generating a 5D vector from an image.


## Training ##
To train the model, just use the following command for both models:

```bash
python VAE/train.py --dataset celeba # Change dataset for MNIST, FashionMNIST, and CIFAR
python VQVAE/train.py --dataset celeba --prior_only False # This is for training VQVAE model without prior
python VQVAE/train.py --dataset celeba --prior_only True # This is for training GatexPixel CNN model with frozen weight of VQVAE
python VAE_pi/train.py
```

## Sampling ##
To generate reconstrcuted/synthesis images, using the following command for both models:
```bash
python VAE_pi/test_img.py # VAE_pi for generating an image with PI
python VAE_pi/test_distibution.py # VAE_pi to check the predicted distribution compared to the dataset
python VAE/test.py --dataset celeba --task generate # Randomly generate dataset from the latent space
python VAE/test.py --dataset celeba --task reconstruct # Reconstruct the dataset
python VQVAE/test.py --dataset celeba --task gen # Randomly generate dataset from the latent space
python VQVAE/test.py --dataset celeba --task recon # Reconstruct the dataset
```


## Result of VAE ##
### Celeba Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/imgss/VAE_celeba_recon.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/imgs/VAE_celeba_gen.png)

### MNIST Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/imgs/VAE_MNIST_recon.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/imgs/VAE_MNIST_gen.png)

### Fashion MNIST Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/imgs/VAE_fashion_recon.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/imgs/VAE_fashion_gen.png)

### CIFAR Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/imgs/VAE_cifar_recon.png)

![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/imgs/VAE_cifar_gen.png)


## Result of VQVAE ##
### Celeba Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/imgs/VQVAE_celeba_recon.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/imgs/VQVAE_celeba_gen.png)

### MNIST Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/imgs/VQVAE_MNIST_recon.png)


![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/imgs/VQVAE_MNIST_gen.png)


### Fashion Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/imgs/VQVAE_fashion_recon.png)


![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/imgs/VQVAE_fashion_gen.png)


### CIFAR Dataset ###
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/imgs/VQVAE_cifar_recon.png)


![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/imgs/VQVAE_cifar_gen.png)