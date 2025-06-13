# DeepLabV3 #
First of all, please follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the cityscapes dataset.


## Training ##
```bash
python train.py --dataset cityspaces --model deeplabv3 --backbone resnet101 --class_num 19
python train.py --dataset voc --model deeplabv3 --backbone resnet101 --class_num 20
```

## Evaluation ##
```bash
python eval.py --dataset cityspaces --model deeplabv3 --backbone resnet101 --class_num 19
python eval.py --dataset voc --model deeplabv3 --backbone resnet101 --class_num 20
```

## Test ##
```bash
python test.py --dataset cityspaces --model deeplabv3 --backbone resnet101 --class_num 19
python test.py --dataset voc --model deeplabv3 --backbone resnet101 --class_num 20
```