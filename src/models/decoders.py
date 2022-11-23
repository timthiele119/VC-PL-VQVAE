from abc import abstractmethod
from src.models.base import BaseModule, ResBlock
import torch
from torch import nn


class Decoder(BaseModule):

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(inputs)

    @abstractmethod
    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
        pass


class ResNetDecoder(Decoder):

    def __init__(self, embedding_dim: int, out_channels: int):
        super(ResNetDecoder, self).__init__()
        self.decoding = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim, out_channels=32, kernel_size=3, stride=1, padding="same"),
            ResBlock(in_channels=32, hidden_channels=64),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            ResBlock(in_channels=32, hidden_channels=64),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding="same")
        )

    @classmethod
    def from_config(cls, config: dict):
        pass

    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoding(inputs)
