from torch.utils.data import DataLoader

from Segmentation_2d.transforms import *
from Segmentation_2d.dataset.cityscapes import CityScapesDataset, cityscapes_class_dict
from Segmentation_2d.dataset.voc import VocSegmentationDataset, voc_class_dict

def get_dataset(args):
    dataset_type = args.dataset
    crop_size = args.crop_size
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    test_batch_size = args.test_batch_size
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if dataset_type == "cityscapes":
        data_path = "Dataset/cityscapes/"
        meta_path = "Dataset/cityscapes/meta"
        
        train_transform = Compose([
            RandomCrop(size=(crop_size, crop_size)),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        
        if args.cityscapes_crop_val:
            val_transform = Compose([
                Resize(args.crop_size),
                CenterCrop(args.crop_size),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        else:
            val_transform = Compose([
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
        
        if args.voc_crop_val:
            val_transform = Compose([
                Resize(args.crop_size),
                CenterCrop(args.crop_size),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        else:
            val_transform = Compose([
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        
        train_dataset = VocSegmentationDataset(root=voc_data_root, year=voc_year, split='train', download=voc_download, transform=train_transform)
        val_dataset = VocSegmentationDataset(root=voc_data_root, year=voc_year, split='val', download=False, transform=val_transform)
        test_dataset = VocSegmentationDataset(root=voc_data_root, year=voc_year, split='val', download=False, transform=val_transform)
        class_dict = voc_class_dict
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
        
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    
    return train_dataloader, val_dataloader, test_dataloader, class_dict, mean, std