import torch.optim as optim

OPTIMIZER_DICT = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
}

def build_optimizer(opts, params):
    optimizer_name = opts.pop('name', None)
    if optimizer_name is None or optimizer_name not in OPTIMIZER_DICT:
        raise ValueError("Optimizer name is not found.")
    optimizer_factory = OPTIMIZER_DICT[optimizer_name]
    if optimizer_name == "Adam":
        opts.betas = tuple(opts.get('betas', None))
        opts.eps = float(opts.get('eps'))
    optimizer = optimizer_factory(params=params, **opts)
    return optimizer