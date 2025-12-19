from Object_detection_2d.utils.config_utils import *
from Object_detection_2d.utils.vis_utils import *

VISUALIZER_DICT = {
    "ImageDetectionVisualizer": ImageDetectionVisualizer
}

def build_visualizer(class_dict, cmap, opts):
    visualizer_name = opts.pop("name", None)
    if visualizer_name is None or visualizer_name not in VISUALIZER_DICT:
        raise ValueError("Visualizer name is not recognized.")
    visualizer_factory = VISUALIZER_DICT[visualizer_name]
    visualizer = visualizer_factory(class_dict=class_dict, cmap=cmap, **opts)
    return visualizer