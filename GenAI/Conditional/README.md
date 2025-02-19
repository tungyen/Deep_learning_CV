# Conditional Diffusion Models

In this folder, I implemented `Classification Free Guidance Diffusion`. It could be trained  on [CIFAR-10 64x64](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution). Any dataset with category is supported for this model as well (Like COCO). You should go to the README of Dataset first for collecting data. In CFDG, DDIM sampler is also supported, you can change to DDPM if you want since DDIM is the default sampler in CFDG. 


In this part, I implement four different diffusion schedules:
1. Linear
2. Cosine
3. Quadratic
4. Sigmoid


# _Training_ #
To train the model, just use the following command for both models:

```bash
python cfdg.py # for Classifier Free Guidance Diffusion
```

# _Sampling_ #
To generate synthesis images, using the following command for both models:
```bash
python cfdg_sampling.py --img_num 10 --sampler "DDIM" --label 5 --img_size 64 # for Classifier Free Guidance Diffusion
```

# _Result of cfdg_ #
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Conditional/images/CFDG_DDIM_cifar_0.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Conditional/images/CFDG_DDIM_cifar_3.png)
