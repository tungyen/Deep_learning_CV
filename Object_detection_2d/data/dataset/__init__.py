from Object_detection_2d.data.dataset.voc import VocDetectionDataset

DATASET_DICT = {
    'VOC': VocDetectionDataset
}

def build_dataset(opts, transform=None, target_transform=None):
    dataset_name = opts.pop('name', None)
    factory = DATASET_DICT[dataset_name]
    dataset = factory(**opts, transform=transform, target_transform=target_transform)
    return dataset