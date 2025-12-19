from core.metrics.confusion_matrix import ConfusionMatrix
from core.metrics.part_seg_metrics import PartSegIOU
from core.metrics.detection_map import DetectionMap

METRICS_DICT = {
    "ConfusionMatrix": ConfusionMatrix,
    "PartSegIOU": PartSegIOU,
    "DetectionMap": DetectionMap
}

def build_metrics(class_dict, dataset, opts):
    metrics_name = opts.pop("name", None)
    if metrics_name is None or metrics_name not in METRICS_DICT:
        raise ValueError(f"Metrics name '{metrics_name}' is not recognized. Available losses: {list(METRICS_DICT.keys())}")
    metrics_factory = METRICS_DICT[metrics_name]
    metrics = metrics_factory(class_dict=class_dict, dataset=dataset, **opts)
    return metrics