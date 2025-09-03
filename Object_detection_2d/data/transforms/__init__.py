from Object_detection_2d.SSD.model.anchors import PriorBox
from Object_detection_2d.data.transforms.transforms import *
from Object_detection_2d.data.transforms.target_transforms import SSDTargetTransformOffset, SSDTargetTransformCoord

TARGET_TRANSFORM_DICT = {
    "Offset": SSDTargetTransformOffset,
    "Coord": SSDTargetTransformCoord,
}

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
    target_transform_name = args['target_transform']
    target_transform_factory = TARGET_TRANSFORM_DICT[target_transform_name]
    transform = target_transform_factory(PriorBox(args)(), args['center_variance'],
                                   args['size_variance'], args['iou_thres'])
    return transform