def setup_args_with_dataset(dataset_type, args):
    if dataset_type == 'modelnet40':
        args.cls_class_num = 40
        args.seg_class_num = 40
        args.n_points = 2048
        args.n_feats = 0
    elif dataset_type == 's3dis':
        args.cls_class_num = 14
        args.seg_class_num = 14
        args.n_points = 8192
        args.n_feats = 6
    return args