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
    elif dataset_type == "shapenet":
        args.cls_class_num = 16
        args.seg_class_num = 50
        args.n_points = 2048
        if args.normal_channel:
            args.n_feats = 3
        else:
            args.n_feats = 0
    else:
        raise ValueError(f'Unknown dataset {dataset_type}.')
    return args