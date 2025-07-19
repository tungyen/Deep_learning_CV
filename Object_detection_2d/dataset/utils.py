from torch.utils.data import DataLoader

from Object_detection_2d.dataset.voc import VocDetectionDataset, voc_id2class
from Object_detection_2d.dataset.transforms import *

def get_dataset(args):
    dataset_type = args.dataset
    crop_size = args.crop_size
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    test_batch_size = args.test_batch_size
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
        
    if dataset_type == "voc":
        voc_data_root = args.voc_data_root
        voc_year = args.voc_year
        voc_download = args.voc_download
        crop_size = args.crop_size
        
        train_transform = Compose([
            ColorJitter(),
            RandomExpand(),
            RandomCrop(),
            RandomHorizontalFlip(),
            Resize(size=(crop_size, crop_size)),
            Normalize(mean=mean, std=std),
        ])
        
        val_transform = Compose([
            Resize(size=(crop_size, crop_size)),
            Normalize(mean=mean, std=std),
        ])
        
        train_dataset = VocDetectionDataset(root=voc_data_root, year=voc_year, split='train', download=voc_download, transform=train_transform)
        val_dataset = VocDetectionDataset(root=voc_data_root, year=voc_year, split='val', download=False, transform=val_transform)
        test_dataset = VocDetectionDataset(root=voc_data_root, year=voc_year, split='val', download=False, transform=val_transform)
        class_dict = voc_id2class
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
        
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                      collate_fn=train_dataset.collate_fn, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False,
                                    collate_fn=val_dataset.collate_fn, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True,
                                     collate_fn=val_dataset.collate_fn, num_workers=4)
    
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    
    return train_dataloader, val_dataloader, test_dataloader, class_dict, mean, std