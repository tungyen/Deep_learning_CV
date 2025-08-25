import yaml

def parse_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config