from Object_detection_2d.SSD.model.ssd import *
from Object_detection_2d.SSD.model.anchors import PriorBox
from Object_detection_2d.SSD.model.layer import L2Norm
from Object_detection_2d.SSD.model.vgg import VGG
from Object_detection_2d.SSD.model.post_processor import PostProcessor

MODEL_DICT = {
    "SSD": SSD
}

def build_model(opts):
    model_name = opts.pop('name', None)
    if model_name is None or model_name not in MODEL_DICT:
        raise ValueError(f"Missing model name or unknown model.")
    model_factory = MODEL_DICT[model_name]
    model = model_factory(**opts)
    return model