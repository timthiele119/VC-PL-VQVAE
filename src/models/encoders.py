from abc import abstractmethod
import torch
from torch import nn
from torch.nn import functional as F
from src.models.modules import ResBlock1d


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encode(inputs)

    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        pass


class LearnedDownsamplingEncoder1d(Encoder):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, kernel_size: int,
                 downsampling_steps: int):
        super(LearnedDownsamplingEncoder1d, self).__init__()
        self.pad = DivisibleByDownsamplingFactorPad1d(2 * downsampling_steps)
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding="same")
        self.downsampling_stack = nn.ModuleList(
            [LearnedDownsamplingResBlock1D(hidden_channels, hidden_channels, kernel_size)
             for _ in range(downsampling_steps)]
        )
        self.conv_out = nn.Conv1d(hidden_channels, out_channels, kernel_size, padding="same")

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        outs = self.pad(inputs)
        outs = self.conv_in(outs)
        for downsampling in self.downsampling_stack:
            outs = downsampling(outs)
        outs = self.conv_out(outs)
        return outs


class LearnedDownsamplingResBlock1D(nn.Module):
    """
    Downsamples a 1D input by a factor of 2 and applies a residual block before and after the downsampling operation
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int):
        super(LearnedDownsamplingResBlock1D, self).__init__()
        self.res_block1 = ResBlock1d(in_channels, hidden_channels, kernel_size)
        self.downsampling = nn.Conv1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.res_block2 = ResBlock1d(in_channels, hidden_channels, kernel_size)

    def forward(self, inputs: torch.Tensor):
        outs = self.res_block1(inputs)
        outs = self.downsampling(outs)
        outs = self.res_block2(outs)
        return outs


class DivisibleByDownsamplingFactorPad1d(nn.Module):
    """"
    Adds as many zero padding elements to the right side of a sequence of arbitrary length such that the resulting sequence is divisible by a
    given downsampling factor
    """
    def __init__(self, downsampling_factor: int):
        super(DivisibleByDownsamplingFactorPad1d, self).__init__()
        self.downsampling_factor = downsampling_factor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        len_seq = inputs.shape[-1]
        right_pad = (len_seq % self.downsampling_factor > 0) * \
                    (self.downsampling_factor - len_seq % self.downsampling_factor)
        pad = (0, right_pad)
        padded_inputs = F.pad(inputs, pad, mode="constant", value=0.0)
        return padded_inputs
