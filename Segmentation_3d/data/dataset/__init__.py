from Segmentation_3d.data.dataset.chair import ChairDataset
from Segmentation_3d.data.dataset.modelnet40 import ModelNet40Dataset
from Segmentation_3d.data.dataset.s3dis import S3disDataset
from Segmentation_3d.data.dataset.shapenet import ShapeNetDataset

DATASET_DICT = {
    'Chair': ChairDataset,
    'ModelNet40': ModelNet40Dataset,
    'S3DIS': S3disDataset,
    'ShapeNet': ShapeNetDataset
}

def build_dataset(opts, transforms=None):
    dataset_name = opts.pop('name', None)
    factory = DATASET_DICT[dataset_name]
    dataset = factory(**opts, transforms=transforms)
    return dataset