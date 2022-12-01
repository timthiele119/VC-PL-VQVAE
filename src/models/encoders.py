from abc import abstractmethod
import torch
from torch import nn
from src.models.base import BaseModule, ResBlock


class Encoder(BaseModule):

    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encode(inputs)

    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        pass


class ResNetEncoder(Encoder):

    def __init__(self, in_channels: int, encoding_dim: int):
        super(ResNetEncoder, self).__init__()
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding="same"),
            ResBlock(in_channels=32, hidden_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResBlock(in_channels=32, hidden_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=encoding_dim, kernel_size=3, stride=1, padding="same")
        )

    @classmethod
    def from_config(cls, config: dict):
        pass

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoding(inputs)
