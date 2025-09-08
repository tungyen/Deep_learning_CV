from Object_detection_2d.CenterNet.model.center_net import CenterNet

MODEL_DICT = {
    "CenterNet": CenterNet,
}

def build_model(args):
    model_config  = args['model']
    model_name = model_config.pop('name', None)
    model_factory = MODEL_DICT.get(model_name, None)
    if model_factory is None:
        raise ValueError(f"Model {model_name} is not supported.")
    else:
        model = model_factory(model_config)
        return model