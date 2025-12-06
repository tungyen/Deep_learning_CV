from torch.utils.data import Dataset, DataLoader
import os
from torch.utils.data.distributed import DistributedSampler
    
def get_dataset(args):
    dataset_type = args.dataset
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    test_batch_size = args.test_batch_size
    n_points = args.n_points
        
    if dataset_type == "modelnet40":
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
        class_list = ['clutter', 'ceiling', 'floor', 'wall', 'beam', 'column', 'door',
               'window', 'table', 'chair', 'sofa', 'bookcase', 'board', 'stairs']
        class_dict = {i: name for i, name in enumerate(class_list)}

    ocal_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size // world_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size // world_size, sampler=val_sampler) 
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size // world_size, sampler=test_sampler)
    return train_dataloader, val_dataloader, test_dataloader, class_dict