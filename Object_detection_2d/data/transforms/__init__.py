from Object_detection_2d.SSD.model.anchors import PriorBox
from Object_detection_2d.data.transforms.transforms import *
from Object_detection_2d.data.transforms.target_transforms import SSDTargetTransform

def build_transforms(args, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(args['img_mean']),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(args['img_size']),
            SubtractMeans(args['img_mean']),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(args['img_size']),
            SubtractMeans(args['img_mean']),
            ToTensor()
        ]
    transform = Compose(transform)
    return transform

def build_target_transform(args):
    transform = SSDTargetTransform(PriorBox(args)(), args['center_variance'],
                                   args['size_variance'], args['iou_thres'])
    return transform