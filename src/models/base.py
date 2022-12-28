from abc import abstractmethod
import torch
from torch import nn


class BaseModule(nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        pass


class Encoder(BaseModule):

    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encode(inputs)

    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        pass


class Decoder(BaseModule):

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(inputs)

    @abstractmethod
    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
        pass


class VectorQuantizer(BaseModule):

    def __init__(self):
        super(VectorQuantizer, self).__init__()

    def forward(self, inputs: torch.Tensor) -> dict:
        return self.quantize(inputs)

    @abstractmethod
    def quantize(self, inputs: torch.Tensor) -> dict:
        pass
