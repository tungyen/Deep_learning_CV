import yaml
from easydict import EasyDict

def parse_config(config_path):
    with open(config_path) as f:
        opts = yaml.safe_load(f)
    opts = EasyDict(opts)
    opts.datasets = EasyDict()

    for split in ['train', 'val', 'test']:
        opts.datasets[split] = EasyDict()
        opts.datasets[split].name = opts.dataset_name
        opts.datasets[split].data_path = opts.data_path
        opts.datasets[split].split = split

        if opts.dataset_name == "Flower":
            if split == 'test':
                opts.datasets[split].split = 'val'
    return opts