from torch.utils.data import Dataset
import os
    
def get_dataset(args):
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    test_batch_size = args.test_batch_size
    n_points = args.n_points

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