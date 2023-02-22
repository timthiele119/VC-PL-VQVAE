import os
from pytorch_lightning.cli import LightningCLI
from src.data.datamodules import VCDataModule
from src.models.hle_vqvae_vc import HleVqVaeVc
from src.params import global_params

os.environ['CUDA_VISIBLE_DEVICES'] = str(global_params.CUDA_VISIBLE_DEVICES)


def cli_main():
    cli = LightningCLI(HleVqVaeVc, VCDataModule)


if __name__ == "__main__":
    cli_main()
