from Segmentation_2d.data.dataset.voc import VocSegmentationDataset
from Segmentation_2d.data.dataset.cityscapes import CityScapesDataset

DATASET_DICT = {
    'VOC': VocSegmentationDataset,
    'CityScapes': CityScapesDataset,
}

def build_dataset(opts, transform=None):
    dataset_name = opts.pop('name', None)
    factory = DATASET_DICT[dataset_name]
    dataset = factory(**opts, transform=transform)
    return dataset