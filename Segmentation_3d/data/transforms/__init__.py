from Segmentation_3d.data.transforms.transforms import *

TRANSFORM_DICT = {
    "NormalizePointClouds": NormalizePointClouds,
    "RandomScalePointClouds": RandomScalePointClouds,
    "RandomShiftPointClouds": RandomShiftPointClouds,
    "RandomRotatePointClouds": RandomRotatePointClouds,
    "RandomJitterPointclouds": RandomJitterPointClouds,
    "FPS": FPS,
    "TransposePointClouds": TransposePointClouds,
    "ToTensor": ToTensor,
}

def build_transforms(opts):
    transforms = []
    for cfg in opts:
        transform_name = cfg.pop('type')
        transform = TRANSFORM_DICT[transform_name](**cfg)
        transforms.append(transform)
    transforms = Compose(transforms)
    return transforms

