import torch.optim as optim

OPTIMIZER_DICT = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
}

def build_param_groups(model, base_lr, lr_mult_dict=None):
    if lr_mult_dict is None:
        return [{"params": model.parameters(), "lr": base_lr}]

    # Initialize a group for each key + one for unmatched params
    groups = {key: {"params": [], "lr": base_lr * mult, "name": key} 
              for key, mult in lr_mult_dict.items()}
    default_group = {"params": [], "lr": base_lr, "name": "default"}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = False
        for key in lr_mult_dict:
            if key in name:
                groups[key]["params"].append(param)
                matched = True
                break
        if not matched:
            default_group["params"].append(param)

    param_groups = [g for g in groups.values() if g["params"]]
    if default_group["params"]:
        param_groups.append(default_group)

    return param_groups

def build_optimizer(opts, model):
    optimizer_name = opts.pop('name', None)
    base_lr = opts.get('lr')
    lr_multi_dict = opts.pop('lr_multi_dict', None)

    if optimizer_name is None or optimizer_name not in OPTIMIZER_DICT:
        raise ValueError("Optimizer name is not found.")
    optimizer_factory = OPTIMIZER_DICT[optimizer_name]

    params = build_param_groups(model, base_lr, lr_multi_dict)
    optimizer = optimizer_factory(params=params, **opts)
    return optimizer
