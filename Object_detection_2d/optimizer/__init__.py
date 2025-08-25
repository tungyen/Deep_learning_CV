import torch.optim as optim

OPTIMIZER_DICT = {
    "SGD": optim.SGD,
    "Adam": optim.Adam
}

def build_optimizer(args, params):
    optimizer_config = args.pop('optimizer', None)
    optimizer_name = optimizer_config.pop('name', None)
    optimizer_factory = OPTIMIZER_DICT[optimizer_name]
    optimizer = optimizer_factory(params=params, **optimizer_config)
    return optimizer
