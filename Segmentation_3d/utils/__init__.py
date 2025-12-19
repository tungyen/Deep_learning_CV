from Segmentation_3d.utils.vis_utils import *
from Segmentation_3d.utils.config_utils import *

VISUALIZER_DICT = {
    "PointCloudSegVisualizer": PointCloudSegVisualizer,
    "PointCloudClsVisualizer": PointCloudSegVisualizer,
}

def build_visualizer(opts):
    visualizer_name = opts.pop('name', None)
    if visualizer_name is None or visualizer_name not in VISUALIZER_DICT:
        raise ValueError("Visualizer type not found.")

    visualizer_factory = VISUALIZER_DICT[visualizer_name]
    return visualizer_factory(**opts)