import argparse
import yaml
from addict import Dict

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./smygw/learning/config.yaml')
args = parser.parse_args()
CONFIG = Dict(yaml.safe_load(open(args.config)))
