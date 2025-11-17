from Segmentation_2d.loss.loss import CrossEntropyLoss, LovaszSoftmaxLoss, BoundaryLoss

LOSS_DICT = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "LovaszSoftmaxLoss": LovaszSoftmaxLoss,
    "BoundaryLoss": BoundaryLoss,
}

def build_loss(opts):
    loss_name = opts.pop("name", None)
    if loss_name is None or loss_name not in LOSS_DICT:
        raise ValueError(f"Loss name '{loss_name}' is not recognized. Available losses: {list(LOSS_DICT.keys())}")
    loss_factory = LOSS_DICT[loss_name]
    criterion = loss_factory(**opts)
    return criterion