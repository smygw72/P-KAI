import argparse
import yaml
from addict import Dict

args = parser.parse_args()
CONFIG = Dict(yaml.safe_load(open(args.config)))
