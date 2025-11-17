from Segmentation_2d.metrics.confusion_matrix import ConfusionMatrix

METRICS_DICT = {
    "ConfusionMatrix": ConfusionMatrix,
}

def build_metrics(opts):
    metrics_name = opts.pop("name", None)
    if metrics_name is None or metrics_name not in METRICS_DICT:
        raise ValueError(f"Metrics name '{metrics_name}' is not recognized. Available losses: {list(METRICS_DICT.keys())}")
    metrics_factory = METRICS_DICT[metrics_name]
    metrics = metrics_factory(**opts)
    return metrics