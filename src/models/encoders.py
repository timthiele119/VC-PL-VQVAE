from abc import abstractmethod
from typing import Any, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from src.models.modules import ResBlock1d, HleConv1d, WaveNetLikeStack


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
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

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        outs = self.pad(seq)
        outs = self.conv_in(outs)
        for downsampling in self.downsampling_stack:
            outs = downsampling(outs)
        outs = self.conv_out(outs)
        return outs


class HleEncoder(Encoder):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            output_latent_dim: int,
            gate_dim: int,
            residual_dim: int,
            skip_dim: int,
            kernel_size: int,
            dilation_steps: int,
            dilation_repeats: int
    ):
        super(HleEncoder, self).__init__()
        self.input_layer = nn.Sequential(
            HleConv1d(in_channels=input_dim, out_channels=2*residual_dim, kernel_size=15),
            nn.GLU(dim=1)
        )
        self.downsampling_layer = nn.Sequential(
            HleConv1d(in_channels=residual_dim, out_channels=2*residual_dim, kernel_size=8, stride=2),
            nn.InstanceNorm1d(2*residual_dim, momentum=0.8),
            nn.GLU(dim=1)
        )
        self.dilation_stack = WaveNetLikeStack(residual_dim=residual_dim, gate_dim=gate_dim, skip_dim=skip_dim,
                                               kernel_size=kernel_size, dilation_steps=dilation_steps,
                                               dilation_repeats=dilation_repeats)
        self.encoding_hidden_dim, self.encoding_latent_dim = output_dim, output_latent_dim
        output_dim = output_dim + output_latent_dim
        self.output_layer = nn.Sequential(
            HleConv1d(in_channels=skip_dim, out_channels=2*output_dim, kernel_size=kernel_size),
            nn.InstanceNorm1d(2*output_dim, momentum=0.8),
            nn.GLU(dim=1),
            HleConv1d(in_channels=output_dim, out_channels=output_dim, kernel_size=1)
        )

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.input_layer(seq)
        seq = self.downsampling_layer(seq)
        seq = self.dilation_stack(seq)
        output = self.output_layer(seq)
        output_latent = output[:, :self.encoding_latent_dim, :]
        output_hidden = output[:, self.encoding_latent_dim:, :] \
            if self.encoding_hidden_dim > 0 else None
        return output_hidden, output_latent


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
