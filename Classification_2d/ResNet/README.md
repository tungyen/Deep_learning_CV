# ResNet #
First you should follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the flower dataset.

Before training, you should download the resnet pretrained weight from PyTorch in a pre-defined folder `pre_model` from [this website](https://download.pytorch.org/models/resnet34-333f7ec4.pth)

## Training ##
```bash
python train.py
```

## Evaluation ##
To be continue

## Testing ##
```bash
python predict.py
```

Now the code only predict the result based on 1 single image. The code for predict a batch of image will be released soon.