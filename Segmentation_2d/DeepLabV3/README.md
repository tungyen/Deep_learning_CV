# DeepLabV3 #
First of all, please follow [My Dataset Guidance](https://github.com/tungyen/Deep_learning_CV/tree/master/Dataset) to download the cityscapes dataset. This part includes DeepLabV3 and DeepLabV3+ model for image semantic segmentation. For the first time use of Paskal VOC dataset, please add argument --voc_download True to download dataset under Dataset folder.To download different year of Paskal VOC dataset, change voc_year in argument.


## Training ##
```bash
# DeepLabV3
python train.py --dataset cityspaces --model deeplabv3 --backbone resnet101 
python train.py --dataset voc --model deeplabv3 --backbone resnet101

# DeepLabV3+
python train.py --dataset cityspaces --model deeplabv3plus --backbone resnet101 
python train.py --dataset voc --model deeplabv3plus --backbone resnet101
```

## Evaluation ##
# DeepLabV3
```bash
python eval.py --dataset cityspaces --model deeplabv3 --backbone resnet101 
python eval.py --dataset voc --model deeplabv3 --backbone resnet101

# DeepLabV3+
python eval.py --dataset cityspaces --model deeplabv3plus --backbone resnet101 
python eval.py --dataset voc --model deeplabv3plus --backbone resnet101
```

## Test ##
# DeepLabV3
```bash
python test.py --dataset cityspaces --model deeplabv3 --backbone resnet101 
python test.py --dataset voc --model deeplabv3 --backbone resnet101

# DeepLabV3+
python test.py --dataset cityspaces --model deeplabv3plus --backbone resnet101 
python test.py --dataset voc --model deeplabv3plus --backbone resnet101
```