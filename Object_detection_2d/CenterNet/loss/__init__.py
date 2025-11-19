from Object_detection_2d.CenterNet.loss.center_net_loss import CenterDetectionLoss

LOSS_DICT = {
    'CenterDetectionLoss': CenterDetectionLoss,
}

def build_loss(opts):
    loss_name = opts.pop('name', None)
    if loss_name is not None and loss_name in LOSS_DICT:
        loss_factory = LOSS_DICT[loss_name]
        loss = loss_factory(**opts)
        return loss
    else:
        raise ValueError(f"Loss name {loss_name} is not supported.")