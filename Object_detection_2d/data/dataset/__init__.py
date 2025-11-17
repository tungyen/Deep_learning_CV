from Object_detection_2d.data.dataset.voc import voc_id2class, VocDetectionDataset

DATASET_DICT = {
    'VOC': VocDetectionDataset
}

def build_dataset(args, stage, device=None, transform=None, target_transform=None, is_train=True):
    dataset_config = args['datasets']
    dataset_name = dataset_config['name']
    factory = DATASET_DICT[dataset_name]
    if factory == VocDetectionDataset:
        dataset_config['keep_difficult'] = not is_train
        if is_train:
            dataset_config['split'] = 'train'
        else:
            dataset_config['split'] = 'val'
    dataset = factory(dataset_config, stage, device, transform, target_transform)
    return dataset