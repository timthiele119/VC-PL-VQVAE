from abc import abstractmethod
from torch import nn


class BaseModule(nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        pass


class ResBlock(BaseModule):
    """
    Implements residual block as suggested in https://arxiv.org/abs/1603.05027
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super(ResBlock, self).__init__()
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
