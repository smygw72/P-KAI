import argparse
import yaml
from addict import Dict

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./smygw/inference_config.yaml')
parser.add_argument('--path', default='../sounds/Z_-hWZetOS0.mp3')
args = parser.parse_args()
CONFIG = Dict(yaml.safe_load(open(args.config)))
