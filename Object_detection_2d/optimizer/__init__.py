import torch.optim as optim

OPTIMIZER_DICT = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
}

def build_optimizer(args, params):
    optimizer_config = args.pop('optimizer', None)
    if optimizer_config == None:
        raise ValueError("Optimizer config is not found.")

    optimizer_name = optimizer_config.pop('name', None)
    if optimizer_name is None:
        raise ValueError("Optimizer name is not found.")
    optimizer_factory = OPTIMIZER_DICT[optimizer_name]
    optimizer = optimizer_factory(params=params, **optimizer_config)
    return optimizer
