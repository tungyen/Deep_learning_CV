from torch.utils.data import Dataset
from torchvision import datasets

class CIFAR10(Dataset):
    def __init__(self, data_path, split, transforms=None, download=False):
        train = split == "train"
        self.dataset = datasets.CIFAR10(
            root=data_path,
            train=train,
            transform=None,
            download=download
        )
        self.transforms = transforms
        self.class_dict = {str(i): name for i, name in enumerate(self.dataset.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transforms is not None:
            img, label = self.transforms(img, label)
        return img, label

    def get_class_dict(self):
        return self.class_dict
