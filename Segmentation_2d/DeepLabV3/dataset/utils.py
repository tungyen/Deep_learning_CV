from torch.utils.data import DataLoader
from transforms import *
from dataset.cityscapes import CityScapesDataset, cityscapes_class_dict
from dataset.voc import VocDataset, voc_class_dict

def get_dataset(args):
    dataset_type = args.dataset
    batch_size = args.batch_size
    crop_size = args.crop_size
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    val_transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std),
    ])
    
    if dataset_type == "cityscapes":
        data_path = "../../Dataset/cityscapes/"
        meta_path = "../../Dataset/cityscapes/meta"
        
        train_transform = Compose([
            RandomCrop(size=(crop_size, crop_size)),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        
        train_dataset = CityScapesDataset(data_path, meta_path, "train", transform=train_transform)
        val_dataset = CityScapesDataset(data_path, meta_path, "val", transform=val_transform)
        test_dataset = CityScapesDataset(data_path, meta_path, "test", transform=val_transform)
        class_dict = cityscapes_class_dict
        
    elif dataset_type == "voc":
        voc_data_root = args.voc_data_root
        voc_year = args.voc_year
        voc_download = args.voc_download
        
        train_transform = Compose([
            RandomScale((0.5, 2.0)),
            RandomCrop(size=(crop_size, crop_size), pad_if_needed=True),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        train_dataset = VocDataset(root=voc_data_root, year=voc_year, split='train', download=voc_download, transform=train_transform)
        val_dataset = VocDataset(root=voc_data_root, year=voc_year, split='val', download=False, transform=val_transform)
        test_dataset = VocDataset(root=voc_data_root, year=voc_year, split='val', download=False, transform=val_transform)
        class_dict = voc_class_dict
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return train_dataloader, val_dataloader, test_dataloader, class_dict, mean, std