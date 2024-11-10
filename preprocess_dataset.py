import argparse
import sys

from covrl.utils.config import Config
from covrl.utils.preprocess import preprocess

if __name__ == "__main__":
    sys.setrecursionlimit(10000)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", required=True, help="Path to the configuration file.")

    args = arg_parser.parse_args(sys.argv[1:])

    config_path = args.config
    conf = Config.from_json(config_path)

    preprocess(conf.load_path, conf.testsuites, conf.save_dir)
