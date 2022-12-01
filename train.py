import argparse
import os
import torch
import yaml

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def parse_args() -> dict:
    """
    Parses the arguments given to Python script
    :return:
        args (dict): Arguments
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("config", help="path to config file in yaml format")
    args = vars(parser.parse_args())
    return args


def parse_config(config_file: str) -> dict:
    """
    Parses the program configuration specified in a yaml file
    :param
        config_file (str): path to yaml config file
    :return:
        config (dict): configuration
    """
    config = None
    with open(config_file) as file:
        content = file.read()
        config = yaml.load(content, Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":
    args = parse_args()
    config_file = args["config"]
    config = parse_config(config_file)
    device = config["device"]

    DEVICE = device if torch.cuda.is_available() else "cpu"
    print(f"Use {DEVICE} device")
