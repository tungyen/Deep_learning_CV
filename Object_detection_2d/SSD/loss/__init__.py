from Object_detection_2d.SSD.loss.iou_loss import *
from Object_detection_2d.SSD.loss.multi_box_loss import MultiBoxesLoss
from Object_detection_2d.SSD.loss.multi_box_iou_loss import MultiBoxesIouLoss
from Object_detection_2d.SSD.model.anchors import PriorBox

LOSS_DICT = {
    "MultiBoxesLoss": MultiBoxesLoss,
    "MultiBoxesIouLoss": MultiBoxesIouLoss,
}

def build_loss(args):
    loss_config = args['loss']
    loss_name = loss_config.pop("name", None)
    loss_factory = LOSS_DICT[loss_name]
    criterion = loss_factory(PriorBox(args)(), **loss_config)
    return criterion