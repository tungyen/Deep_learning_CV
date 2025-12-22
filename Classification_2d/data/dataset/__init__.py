from Classification_2d.data.dataset.flower import FlowerDataset
from Classification_2d.data.dataset.cifar_10 import CIFAR10
from Classification_2d.data.dataset.cifar_100 import CIFAR100

DATASET_DICT = {
    'Flower': FlowerDataset,
    "CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
}

def build_dataset(opts, transforms=None):
    dataset_name = opts.pop('name', None)
    factory = DATASET_DICT[dataset_name]
    dataset = factory(**opts, transforms=transforms)
    return dataset