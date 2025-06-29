from torch.utils.data import Dataset, DataLoader, random_split
import os

from Segmentation_3d.dataset.chair import ChairDataset
from Segmentation_3d.dataset.modelnet40 import ModelNet40Dataset
from Segmentation_3d.dataset.s3dis import S3disDataset
from Segmentation_3d.dataset.shapenet import ShapeNetClsDataset, ShapeNetSegDataset

def split_dataset_train_val(dataset: Dataset, split=0.9):
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
    
def get_dataset(args):
    dataset_type = args.dataset
    batch_size = args.batch_size
    n_points = args.n_points
    
    if dataset_type == "chair":
        path = os.path.join("Dataset", "Chair_dataset")
        train_dataset = ChairDataset(path, n_points=n_points)
        train_dataset, val_dataset = split_dataset_train_val(train_dataset)
        test_dataset = ChairDataset(path, train=False, n_points=n_points)
        class_dict = {
            0: "Armrest",
            1: "Backrest",
            2: "Chair legs",
            3: "Cushion"
        }
        
    elif dataset_type == "modelnet40":
        path = os.path.join("Dataset", "ModelNet40_npz")
        train_dataset = ModelNet40Dataset(path, n_points, "train")
        class_dict = train_dataset.id2name
        train_dataset, val_dataset = split_dataset_train_val(train_dataset)
        test_dataset = ModelNet40Dataset(path, n_points, "test")
        
    elif dataset_type == 's3dis':
        path = os.path.join("Dataset", "S3DIS_npz")
        test_area = args.test_area
        max_dropout = args.max_dropout
        block_type = args.block_type
        block_size = args.block_size
        
        train_dataset = S3disDataset(path, "train", test_area, n_points, max_dropout, block_type, block_size)
        train_dataset, val_dataset = split_dataset_train_val(train_dataset)
        test_dataset = S3disDataset(path, "test", test_area, n_points, max_dropout, block_type, block_size)
        class_dict = ['clutter', 'ceiling', 'floor', 'wall', 'beam', 'column', 'door',
               'window', 'table', 'chair', 'sofa', 'bookcase', 'board', 'stairs']

    elif dataset_type == 'shapenet_cls':
        class_choice = args.class_choice
        normal_channel = args.normal_channel
        path = os.path.join("Dataset", "ShapeNetPart")
        train_dataset = ShapeNetClsDataset(root=path, n_points=n_points, split="train", class_choice=class_choice, normal_channel=normal_channel)
        val_dataset = ShapeNetClsDataset(root=path, n_points=n_points, split="val", class_choice=class_choice, normal_channel=normal_channel)
        test_dataset = ShapeNetClsDataset(root=path, n_points=n_points, split="test", class_choice=class_choice, normal_channel=normal_channel)
        class_dict = train_dataset.class2label

    elif dataset_type == 'shapenet_seg':
        class_choice = args.class_choice
        normal_channel = args.normal_channel
        path = os.path.join("Dataset", "ShapeNetPart")
        train_dataset = ShapeNetSegDataset(root=path, n_points=n_points, split="train", class_choice=class_choice, normal_channel=normal_channel)
        val_dataset = ShapeNetSegDataset(root=path, n_points=n_points, split="val", class_choice=class_choice, normal_channel=normal_channel)
        test_dataset = ShapeNetSegDataset(root=path, n_points=n_points, split="test", class_choice=class_choice, normal_channel=normal_channel)
        class_dict = train_dataset.seg_category
    else:
        raise ValueError(f'unknown dataset {dataset_type}')
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader, class_dict