from Object_detection_2d.SSD.model.anchors import PriorBox
from Object_detection_2d.data.transforms.transforms import *
from Object_detection_2d.data.transforms.target_transforms import SSDTargetTransformOffset, SSDTargetTransformCoord, CenterNetTargetTransform

def ssd_transforms(args, is_train):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(args['img_mean']),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            ResizeImg(args['img_size']),
            SubtractMeans(args['img_mean']),
            ToTensor(),
        ]
    else:
        transform = [
            Resize(args['img_size']),
            SubtractMeans(args['img_mean']),
            ToTensor()
        ]
    return transform

def center_net_transforms(args, is_train):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            RandomSampleCrop(),
            RandomMirror(),
            ResizeImgBoxes(args['img_size']),
            Normalize(args['img_mean'], args['img_std']),
            ToTensor(),
        ]
    else:
        transform = [
            ResizeImg(args['img_size']),
            Normalize(args['img_mean'], args['img_std']),
            ToTensor()
        ]
    return transform

TRANSFORM_DICT = {
    "SSD": ssd_transforms,
    "CenterNet": center_net_transforms
}

TARGET_TRANSFORM_DICT = {
    "Offset": SSDTargetTransformOffset,
    "Coord": SSDTargetTransformCoord,
    "CenterNet": CenterNetTargetTransform
}

def build_transforms(args, is_train=True):
    model_name = args['model']['name']
    transform = TRANSFORM_DICT[model_name](args, is_train)
    transform = Compose(transform)
    return transform

def build_target_transform(args):
    target_transform_name = args['target_transform']
    target_transform_factory = TARGET_TRANSFORM_DICT[target_transform_name]
    if target_transform_name != "CenterNet":
        transform = target_transform_factory(PriorBox(args)(), args)
    else:
        transform = target_transform_factory(args)
    return transform