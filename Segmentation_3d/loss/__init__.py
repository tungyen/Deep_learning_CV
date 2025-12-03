from Segmentation_3d.loss.loss import CrossEntropyLoss, FocalLoss, LovaszSoftmaxLoss

LOSS_DICT = {
    "CrossEntropyLoss": CrossEntropyLoss,
    "LovaszSoftmaxLoss": LovaszSoftmaxLoss,
    "FocalLoss": FocalLoss,
}

def build_loss(opts):
    loss_name = opts.pop("name", None)
    if loss_name is None or loss_name not in LOSS_DICT:
        raise ValueError(f"Loss name '{loss_name}' is not recognized. Available losses: {list(LOSS_DICT.keys())}")
    loss_factory = LOSS_DICT[loss_name]
    criterion = loss_factory(**opts)
    return criterion