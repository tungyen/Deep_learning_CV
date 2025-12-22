from Classification_2d.loss.cross_entropy_loss import CrossEntropyLoss

LOSS_DICT = {
    "CrossEntropyLoss": CrossEntropyLoss,
}

def build_loss(opts):
    loss_name = opts.pop("name", None)
    if loss_name is None or loss_name not in LOSS_DICT:
        raise ValueError(f"Loss name '{loss_name}' is not recognized. Available losses: {list(LOSS_DICT.keys())}")
    loss_factory = LOSS_DICT[loss_name]
    criterion = loss_factory(**opts)
    return criterion