from Segmentation_2d.data.transforms.transforms import *

TRANSFORM_DICT = {
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "RandomVerticalFlip": RandomVerticalFlip,
    "CenterCrop": CenterCrop,
    "RandomScale": RandomScale,
    "Scale": Scale,
    "RandomRotation": RandomRotation,
    "Pad": Pad,
    "ToTensor": ToTensor,
    "Normalize": Normalize,
    "RandomCrop": RandomCrop,
    "ColorJitter": ColorJitter,
    "Resize": Resize,
}


def build_transforms(opts):
    transforms = []
    for cfg in opts:
        transform_name = cfg.pop('type')
        transform = TRANSFORM_DICT[transform_name](**cfg)
        transforms.append(transform)
    transforms = Compose(transforms)
    return transforms