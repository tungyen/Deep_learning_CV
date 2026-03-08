from Object_detection_2d.model.SSD import SSD, PriorBox
from Object_detection_2d.model.CenterNet import CenterNet

MODEL_DICT = {
    "SSD": SSD,
    "CenterNet": CenterNet,
}

def build_model(opts):
    model_name = opts.pop('name', None)
    if model_name is None or model_name not in MODEL_DICT:
        raise ValueError(f"Missing model name or unknown model.")
    model_factory = MODEL_DICT[model_name]
    model = model_factory(**opts)
    return model