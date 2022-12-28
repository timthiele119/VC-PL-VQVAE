from src.models.base import BaseModule
import torch
from torch import nn
from torch.nn import functional as F


class ResBlock2d(BaseModule):
    """
    Implements residual block as suggested in https://arxiv.org/abs/1603.05027
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super(ResBlock2d, self).__init__()
        self.transformation = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size, padding="same")
        )

    def forward(self, x):
        return x + self.transformation(x)

    @classmethod
    def from_config(cls, config):
        pass


class ResBlock1d(nn.Module):
    """
    Implements residual block for 1d inputs as suggested in https://arxiv.org/abs/1603.05027
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super(ResBlock1d, self).__init__()
        self.transformation = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, in_channels, kernel_size, padding="same"),
        )

    def forward(self, inputs):
        return inputs + self.transformation(inputs)
