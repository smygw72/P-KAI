# copy from https://zenn.dev/kwashizzz/articles/ml-hydra-param
import argparse
import sys
import os
import yaml
from addict import Dict

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/default.yaml')
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    config.path = args.config
    return config

CONFIG = get_config()
