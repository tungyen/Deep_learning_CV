# DDPM
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/DDPM/images/test.png)


In this folder, I implemented `Denoising Diffusion Probabilistic Model` and `Classifier Free Difussion Guidance`. The former one is trained on the [Landscape Dataset](https://www.kaggle.com/datasets/arnaud58/landscape-pictures). Optionally, any dataset could be acceptable as long as the structure is `datasets/<class_name>`. And the later one is based on [CIFAR-10 64x64](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution). Any dataset with category is supported for this model as well (Like COCO). You should go to the README of Dataset first for collecting data.


# _Training_ #
To train the model, just use the following command for both models:

```bash
python ddpm.py # for DDPM
python cfdg.py # for Classifier Free Guidance Diffusion
```

# _Sampling_ #
To generate synthesis images, using the following command for both models:
```bash
python ddpm_sampling.py # for DDPM
python cfdg_sampling.py # for Classifier Free Guidance Diffusion
```
