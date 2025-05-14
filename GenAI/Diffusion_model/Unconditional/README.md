# Unconditional Diffusion Models #

In this folder, I implemented `Denoising Diffusion Probabilistic Model`, `Denoising Diffusion Implicit Model`, and `Improved Denoising Diffusion Model`. All models could be trained on the [Landscape Dataset](https://www.kaggle.com/datasets/arnaud58/landscape-pictures), or the  [Huggan/smithsonian_butterflies_subset](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset). Optionally, any dataset could be acceptable. You should go to the README of Dataset first for collecting data.


In this part, I implement four different diffusion schedules:
1. Linear
2. Cosine
3. Quadratic
4. Sigmoid


## Training ##
To train the model, just use the following command for both models:

```bash
python ddpm.py # for DDPM
python ddim.py # Note that you can directly use the trained ckpt of ddpm on this model without retraining
python iddpm.py # for iddpm
```

## Sampling ##
To generate synthesis images, using the following command for both models:
```bash
python ddpm_sampling.py # for DDPM
python ddim_sampling.py # for DDIM
python iddpm_sampling.py # for iDDPM
```

Currently, IDDPM is still in progress due to some unknown problem.....

## Result of ddpm ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Diffusion_model/Unconditional/images/DDPM_butterfly.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Diffusion_model/Unconditional/images/DDPM_landscape.png)

## Result of ddim ##
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Diffusion_model/Unconditional/images/DDIM_butterfly.png)
![image](https://github.com/tungyen/Deep_learning_CV/blob/master/GenAI/Diffusion_model/Unconditional/images/DDIM_landscape.png)