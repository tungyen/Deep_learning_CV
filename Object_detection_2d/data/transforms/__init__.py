from Object_detection_2d.SSD.model.anchors import PriorBox
from Object_detection_2d.data.transforms.transforms import *
from Object_detection_2d.data.transforms.target_transforms import *

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
    "ConvertFromInts": ConvertFromInts,
    "PhotometricDistort": PhotometricDistort,
    "Expand": Expand,
    "RandomSampleCrop": RandomSampleCrop,
    "RandomMirror": RandomMirror,
    "ToPercentCoords": ToPercentCoords,
    "ToAbsoluteCoords": ToAbsoluteCoords,
    "ResizeImg": ResizeImg,
    "SubtractMeans": SubtractMeans,
    "ToTensor": ToTensor,
    "Normalize": Normalize,
    "ResizeImgBoxes": ResizeImgBoxes
}

TARGET_TRANSFORM_DICT = {
    "Offset": SSDTargetTransformOffset,
    "Coord": SSDTargetTransformCoord,
    "CenterNet": CenterNetTargetTransform
}

def build_transforms(opts):
    transforms = []
    for cfg in opts:
        transform_name = cfg.pop('type')
        transform_factory = TRANSFORM_DICT[transform_name]
        transforms.append(transform_factory(**cfg))
    transforms = Compose(transforms)
    return transforms

def build_target_transform(opts):
    target_transforms = []
    for cfg in opts:
        target_transform_name = cfg.pop('type')
        target_transform_factory = TARGET_TRANSFORM_DICT[target_transform_name]
        if target_transform_name != "CenterNet":
            prior_cfg = cfg.pop('prior', None)
            if prior_cfg is None:
                raise ValueError("prior cfg in SSD target transform should not be None.")
            target_transform = target_transform_factory(PriorBox(**prior_cfg)(), **cfg)
        else:
            target_transform = target_transform_factory(**cfg)
        target_transforms.append(target_transform)
    target_transforms = Compose_target(target_transforms)
    return target_transforms