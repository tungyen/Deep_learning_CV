from Segmentation_3d.metrics.confusion_matrix import ConfusionMatrix
from Segmentation_3d.metrics.part_seg_metrics import compute_pcloud_partseg_metrics

METRICS_DICT = {
    "ConfusionMatrix": ConfusionMatrix,
    "PartSegMetrics": compute_pcloud_partseg_metrics,
}

def build_metrics(opts):
    metrics_name = opts.pop("name", None)
    if metrics_name is None or metrics_name not in METRICS_DICT:
        raise ValueError(f"Metrics name '{metrics_name}' is not recognized. Available losses: {list(METRICS_DICT.keys())}")
    metrics_factory = METRICS_DICT[metrics_name]
    metrics = metrics_factory(**opts)
    return metrics