import argparse
import glob
import yaml
from addict import Dict


def get_config(inference_mode=False):
    if inference_mode is False:
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='./config/default.yaml')
        parser.add_argument('--tuning', action='store_true')
        args = parser.parse_args()
        config = Dict(yaml.safe_load(open(args.config)))
        config.tuning = args.tuning
    else:
        config_file = glob.glob("./model/**/*.yaml")
        if len(config_file) == 0:
            print("Not found for config file")
        else:
            print(f"config path: {config_file[0]}")
            config = Dict(yaml.safe_load(open(config_file[0])))
    return config
