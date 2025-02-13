# DDPM

In this folder, I implemented `Denoising Diffusion Probabilistic Model`, `Denoising Diffusion Implicit Model`, `Improved Denoising Diffusion Model`, and `Classifier Free Difussion Guidance`. The first three models could be  trained on the [Landscape Dataset](https://www.kaggle.com/datasets/arnaud58/landscape-pictures), or the  [Huggan/smithsonian_butterflies_subset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset). Optionally, any dataset could be acceptable. And the last one is based on [CIFAR-10 64x64](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution). Any dataset with category is supported for this model as well (Like COCO). You should go to the README of Dataset first for collecting data. In CFDG, DDIM sampler is also supported, you can change to DDPM if you want since DDIM is the default sampler in CFDG. 


In this part, I implement four different diffusion schedules:
1. Linear
2. Cosine
3. Quadratic
4. Sigmoid


# _Training_ #
To train the model, just use the following command for both models:

```bash
python ddpm.py # for DDPM
python ddim.py # Note that you can directly use the trained ckpt of ddpm on this model without retraining
python iddpm.py # for iddpm
python cfdg.py # for Classifier Free Guidance Diffusion
```

# _Sampling_ #
To generate synthesis images, using the following command for both models:
```bash
python ddpm_sampling.py # for DDPM
python ddim_sampling.py # for DDIM
python iddpm_sampling.py # for iDDPM
python cfdg_sampling.py --img_num 10 --sampler "DDIM" --label 5 --img_size 64 # for Classifier Free Guidance Diffusion
```

Note that IDDPM is still in progress.....

# _Result of ddpm_ #
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/DDPM/images/DDPM_butterfly.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/DDPM/images/DDPM_landscape.png)

# _Result of ddim_ #
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/DDPM/images/DDIM_butterfly.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/DDPM/images/DDIM_landscape.png)

# _Result of cfdg_ #
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/DDPM/images/CFDG_DDIM_cifar_0.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/DDPM/images/CFDG_DDIM_cifar_3.png)
