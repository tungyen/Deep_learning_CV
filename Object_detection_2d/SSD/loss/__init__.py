from Object_detection_2d.SSD.loss.iou_loss import *
from Object_detection_2d.SSD.loss.multi_box_loss import MultiBoxesLoss
from Object_detection_2d.SSD.loss.multi_box_iou_loss import MultiBoxesIouLoss
from Object_detection_2d.SSD.model.anchors import PriorBox

LOSS_DICT = {
    "MultiBoxesLoss": MultiBoxesLoss,
    "MultiBoxesIouLoss": MultiBoxesIouLoss,
}

def build_loss(opts):
    loss_name = opts.pop("name", None)
    loss_factory = LOSS_DICT[loss_name]
    prior_cfg = opts.pop('prior', None)
    if prior_cfg is None:
        raise ValueError("prior cfg in SSD loss should not be None.")
    criterion = loss_factory(PriorBox(**prior_cfg)(), **opts)
    return criterion