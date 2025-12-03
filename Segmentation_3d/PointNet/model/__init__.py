from Segmentation_3d.PointNet.model.pointnet import PointNetCls, PointNetSemseg, PointNetPartseg
from Segmentation_3d.PointNet.model.pointnet_plus import PointNetPlusCls, PointNetPlusSemseg, PointNetPlusPartseg

MODEL_DICT = {
    "PointNetCls": PointNetCls,
    "PointNetSemseg": PointNetSemseg,
    "PointNetPartseg": PointNetPartseg,
    "PointNetPlusCls": PointNetPlusCls,
    "PointNetPlusSemseg": PointNetPlusSemseg,
    "PointNetPlusPartseg": PointNetPlusPartseg
}

def build_model(opts):
    model_name = opts.pop('name', None)
    if model_name is None or model_name not in MODEL_DICT:
        raise ValueError(f"Model name {model_name} is not valid.")
    model_factory = MODEL_DICT[model_name]
    model = model_factory(**opts)
    return model
