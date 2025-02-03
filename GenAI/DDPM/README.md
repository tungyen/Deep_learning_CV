# DDPM

In this folder, I implemented `Denoising Diffusion Probabilistic Model`, `Denoising Diffusion Implicit Model`, and `Classifier Free Difussion Guidance`. The first two ones are trained on the [Landscape Dataset](https://www.kaggle.com/datasets/arnaud58/landscape-pictures). Optionally, any dataset could be acceptable as long as the structure is `datasets/<class_name>`. And the last one is based on [CIFAR-10 64x64](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution). Any dataset with category is supported for this model as well (Like COCO). You should go to the README of Dataset first for collecting data.


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
python cfdg.py # for Classifier Free Guidance Diffusion
```

# _Sampling_ #
To generate synthesis images, using the following command for both models:
```bash
python ddpm_sampling.py # for DDPM
python ddim_sampling.py # for DDIM
python cfdg_sampling.py # for Classifier Free Guidance Diffusion
```

# _Result of ddpm_ #
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/DDPM/images/ddpm_res.png)

# _Result of ddpm_ #
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/DDPM/images/ddim_res.png)

# _Result of cfdg_ #
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/DDPM/images/cfdg_res.png)
