from Object_detection_2d.SSD.model.ssd import *
from Object_detection_2d.SSD.model.anchors import PriorBox
from Object_detection_2d.SSD.model.layer import L2Norm
from Object_detection_2d.SSD.model.vgg import VGG
from Object_detection_2d.SSD.model.post_processor import PostProcessor

MODEL_DICT = {
    "SSD": SSD
}

def build_model(args):
    model_factory = MODEL_DICT[args['model']]
    model = model_factory(args)
    return model