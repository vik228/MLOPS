from functools import reduce

import yaml
from operator import getitem


def recursive_get(d, *keys):
    if isinstance(keys[0], str) and "." in keys[0]:
        return recursive_get(d, *keys[0].split("."))
    try:
        return reduce(getitem, keys, d)
    except Exception:
        return {}


def load_config_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_config(config_path, *keys):
    config = load_config_params(config_path)
    return recursive_get(config, *keys)
