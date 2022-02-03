import argparse
import sys
import os
import glob
import yaml
from addict import Dict

def get_config(test_mode=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/default.yaml')
    parser.add_argument('--tuning', action='store_true')
    args = parser.parse_args()
    if test_mode is False:
        config = Dict(yaml.safe_load(open(args.config)))
    else:
        config_file = glob.glob("./model/**/*.yaml")[0]
        config = Dict(yaml.safe_load(open(config_file)))
    config.tuning = args.tuning
    return config
