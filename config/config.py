# copy from https://zenn.dev/kwashizzz/articles/ml-hydra-param
import argparse
import sys
import os
from hydra import compose, initialize_config_dir


class Config():
    """
    hydraによる設定値の取得 (conf)
    """
    @staticmethod
    def get_cnf():
        """
        設定値の辞書を取得
        @return
            cnf: OmegaDict
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='default.yaml')
        args = parser.parse_args()

        conf_dir = os.path.join(os.getcwd(), "./config/")
        if not os.path.isdir(conf_dir):
            print(f"Can not find file: {conf_dir}.")
            sys.exit(-1)

        with initialize_config_dir(config_dir=conf_dir):
            cnf = compose(config_name=f"{args.config}")
            return cnf


CONFIG = Config.get_cnf()
