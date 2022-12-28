import argparse
import os
import torch
from torchaudio.transforms import MuLawDecoding
import yaml
from src.data.datasets import VCTKDataset
from src.models.vqvae_vc import VQVAEVC
from torch.utils.data import DataLoader

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

    vctk_audio_dir = "/home/franky3er/Documents/code/data/VCTK/wav48_silence_trimmed"
    vctk_speaker_info_path = "/home/franky3er/Documents/code/data/VCTK/speaker-info.txt"
    root = "data/VCTK"
    sr = 22050

    dataset = VCTKDataset(root, sr, 1024, 4096, vctk_speaker_info_path, vctk_audio_dir)
    dataloader = DataLoader(dataset, batch_size=64)
    mu_encoded_sequences, speakers = next(iter(dataloader))

    mu_law_decoding = MuLawDecoding(256)

    sequences = mu_law_decoding(mu_encoded_sequences).to(device)
    print(sequences.shape)

    vqvae_vc = VQVAEVC(32, 4, 32, 1000, 32, 110).to(device)
    print(vqvae_vc(sequences).shape)


