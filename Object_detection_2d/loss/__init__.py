from Object_detection_2d.loss.iou_loss import *
from Object_detection_2d.loss.multi_box_loss import MultiBoxesLoss
from Object_detection_2d.loss.multi_box_iou_loss import MultiBoxesIouLoss
from Object_detection_2d.loss.center_net_loss import CenterDetectionLoss

from Object_detection_2d.model.SSD import PriorBox

LOSS_DICT = {
    "MultiBoxesLoss": MultiBoxesLoss,
    "MultiBoxesIouLoss": MultiBoxesIouLoss,
    'CenterDetectionLoss': CenterDetectionLoss,
}

def build_loss(opts):
    loss_name = opts.pop("name", None)
    loss_factory = LOSS_DICT[loss_name]
    prior_cfg = opts.pop('prior', None)

    if prior_cfg is None and loss_name in ["MultiBoxesLoss", "MultiBoxesIouLoss"]:
        raise ValueError("prior cfg in SSD loss should not be None.")
    
    if prior_cfg is None:
        criterion = loss_factory(**opts)
    else:
        criterion = loss_factory(PriorBox(**prior_cfg)(), **opts)
    return criterion