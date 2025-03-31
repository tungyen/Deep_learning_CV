# Variational Autoencoder

In this folder, I implemented `Variational Autoencoder` and `Vector Quantization Variational Autoencoder`. And `Vector Quantization Variational Autoencoder` applies `Gated PixelCNN` as the prior generation model for sampling. All models could be trained on the [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), MNIST, Fashion MNIST, and CIFAR dataset.

Note that VAE_pi is a project from an company online assessment. The goal is specifically for generating a 5D vector from an image.


# _Training_ #
To train the model, just use the following command for both models:

```bash
python VAE/train.py --dataset celeba # Change dataset for MNIST, FashionMNIST, and CIFAR
python VQVAE/train.py --dataset celeba --prior_only False # This is for training VQVAE model without prior
python VQVAE/train.py --dataset celeba --prior_only True # This is for training GatexPixel CNN model with frozen weight of VQVAE
python VAE_pi/train.py
```

# _Sampling_ #
To generate reconstrcuted/synthesis images, using the following command for both models:
```bash
python VAE_pi/test_img.py # VAE_pi for generating an image with PI
python VAE_pi/test_distibution.py # VAE_pi to check the predicted distribution compared to the dataset
python VAE/test.py --dataset celeba --task generate # Randomly generate dataset from the latent space
python VAE/test.py --dataset celeba --task reconstruct # Reconstruct the dataset
python VQVAE/test.py --dataset celeba --task gen # Randomly generate dataset from the latent space
python VQVAE/test.py --dataset celeba --task recon # Reconstruct the dataset
```


# _Result of Vanila VAE_ #
## _Celeba Dataset_ ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/img/VAE_celeba_recon.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/img/VAE_celeba_gen.png)

## _MNIST Dataset_ ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/img/VAE_MNIST_recon.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/img/VAE_MNIST_gen.png)

## _Fashion MNIST Dataset_ ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/img/VAE_fashion_recon.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/img/VAE_fashion_gen.png)

## _CIFAR Dataset_ ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/img/VAE_cifar_recon.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VAE/img/VAE_cifar_gen.png)


# _Result of VQVAE_ #
## _Celeba Dataset_ ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/img/VQVAE_celeba_recon.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/img/VQVAE_celeba_gen.png)

## _MNIST Dataset_ ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/img/VQVAE_MNIST_recon.png)


![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/img/VQVAE_MNIST_gen.png)


## _Fashion Dataset_ ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/img/VQVAE_fashion_recon.png)


![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/img/VQVAE_fashion_gen.png)


## _CIFAR Dataset_ ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/img/VQVAE_cifar_recon.png)


![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Autoencoder/VQVAE/img/VQVAE_cifar_gen.png)