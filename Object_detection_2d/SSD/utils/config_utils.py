import yaml
from easydict import EasyDict

def parse_config(config_path):
    with open(config_path) as f:
        opts = yaml.safe_load(f)

    opts = EasyDict(opts)
    for split in ['train', 'val', 'test']:
        opts.datasets[split].name = opts.datasets.dataset_name
        opts.datasets[split].data_path = opts.datasets.data_path
    return opts