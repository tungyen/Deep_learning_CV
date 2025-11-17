import yaml
from easydict import EasyDict

def parse_config(config_path):
    with open(config_path) as f:
        opts = yaml.safe_load(f)
    opts = EasyDict(opts)
    opts.metrics.class_num = opts.class_num
    opts.metrics.ignore_index = opts.ignore_index
    opts.model.class_num = opts.class_num
    opts.datasets = EasyDict()

    for split in ['train', 'val', 'test']:
        opts.datasets[split] = EasyDict()
        opts.datasets[split].name = opts.dataset_name
        opts.datasets[split].data_path = opts.data_path
        opts.datasets[split].split = split
        if opts.dataset_name == "CityScapes":
            opts.datasets[split].meta_path = opts.meta_path
        elif opts.dataset_name == "VOC":
            opts.datasets[split].year = opts.year
            opts.datasets[split].download = opts.download

            if split == 'test':
                opts.datasets[split].split = 'val'
        else:
            raise ValueError("Unsupported dataset: {}".format(opts.dataset_name))
    return opts