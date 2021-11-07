# copy from https://zenn.dev/kwashizzz/articles/ml-hydra-param
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
        conf_dir = os.path.join(os.getcwd(), "./")
        if not os.path.isdir(conf_dir):
            print(f"Can not find file: {conf_dir}.")
            sys.exit(-1)

        with initialize_config_dir(config_dir=conf_dir):
            cnf = compose(config_name="config.yaml")
            return cnf


CONFIG = Config.get_cnf()
