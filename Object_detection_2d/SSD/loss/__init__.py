from Object_detection_2d.SSD.loss.iou_loss import *
from Object_detection_2d.SSD.loss.multi_box_loss import MultiBoxesLoss

LOSS_DICT = {
    "MultiBoxesLoss": MultiBoxesLoss
}

def build_loss(args):
    loss_config = args['loss']
    loss_name = loss_config.pop("name", None)
    loss_factory = LOSS_DICT[loss_name]
    criterion = loss_factory(**loss_config)
    return criterion