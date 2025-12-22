from Classification_2d.data.dataset.flower import FlowerDataset

DATASET_DICT = {
    'Flower': FlowerDataset
}

def build_dataset(opts, transforms=None):
    dataset_name = opts.pop('name', None)
    factory = DATASET_DICT[dataset_name]
    dataset = factory(**opts, transforms=transforms)
    return dataset