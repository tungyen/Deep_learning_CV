from Segmentation_2d.model.DeepLabV3 import *
from Segmentation_2d.model.SegFormer import SegFormer
from Segmentation_2d.model.Segmenter import Segmenter


MODEL_DICT = {
    "DeepLabV3": DeepLabV3,
    "DeepLabV3Plus": DeepLabV3Plus,
    "SegFormer": SegFormer,
    "Segmenter": Segmenter,
}

def build_model(opts):
    model_name = opts.pop('name', None)
    if model_name is None or model_name not in MODEL_DICT:
        raise ValueError(f"Model name '{model_name}' is not recognized. Available models: {list(MODEL_DICT.keys())}")
    model_factory = MODEL_DICT[model_name]
    model = model_factory(**opts)
    return model