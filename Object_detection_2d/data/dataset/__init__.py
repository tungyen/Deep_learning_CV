from Object_detection_2d.data.dataset.voc import VocDetectionDataset, voc_cmap

DATASET_DICT = {
    'VOC': VocDetectionDataset
}

CMAP_DICT = {
    'VOC': voc_cmap
}

def build_cmap(opts):
    dataset_name = opts.datasets.dataset_name
    factory = CMAP_DICT[dataset_name]
    return factory

def build_dataset(opts, transform=None, target_transform=None):
    dataset_name = opts.pop('name', None)
    factory = DATASET_DICT[dataset_name]
    dataset = factory(**opts, transform=transform, target_transform=target_transform)
    return dataset