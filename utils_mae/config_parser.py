import os
from pprint import pprint, pformat
import yaml


class Config:
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def parse_yaml(yaml_path):
    with open(yaml_path, 'r',encoding='utf-8') as fin:
        cfg = yaml.load(fin.read(), Loader=yaml.FullLoader)
    return cfg


def get_config(config_file):
    yaml_config = parse_yaml(config_file)
    pprint(yaml_config)
    print("Please check the above information for the configurations", flush=True)
    return Config(yaml_config)

