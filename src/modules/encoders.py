from abc import abstractmethod
from typing import Any, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.modules import ResBlock1d, HleConv1d, WaveNetLikeStack, DivisibleByDownsamplingFactorPad1d


class Encoder(nn.Module):
    """
    Encoder base module
    """

    def __init__(self):
        super(Encoder, self).__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass


class HleEncoder(Encoder):
    """
    Encoder component of the HLE-VQVAE architecture (https://www.isca-speech.org/archive/vccbc_2020/ho20_vccbc.html)

    Args:
        input_dim (int): Number of channels in the input sequence
        output_dim (int): Number of channels in the output sequence
        output_latent_dim (int): Number of channels of the latent sequence (which will be forwarded to the vector
                                 quantizer)
        gate_dim (int): Number of channels for the gating mechanism of the WaveNet-like component
        residual_dim (int): Number of residual channels of the WaveNet-like component
        skip_dim (int): Number of skip channels of the WaveNet-like component
        kernel_size (int): Size of the kernel
        dilation_steps (int): Number of dilation steps of the WaveNet-like component
        dilation_repeats (int): Number of times the dilations are repeated
    """

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


class LearnedDownsamplingEncoder1d(Encoder):
    """
    Simple encoder that downsamples a sequence using residual blocks

    Args:
        in_channels (int): Number of channels in the input sequence
        hidden_channels (int): Number of channels in the intermediate steps
        out_channels (int): Number of channels in the output sequence
        kernel_size (int): Size of the kernel
        downsampling_steps (int): Number of downsampling steps, whereby each downsampling step halves the input sequence
    """

    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            kernel_size: int,
            downsampling_steps: int
    ):
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


class LearnedDownsamplingResBlock1D(nn.Module):
    """
    Downsamples a 1D input by a factor of 2 and applies a residual block before and after the downsampling operation

    Args:
        in_channels (int): Number of channels in the input sequence
        hidden_channels (int): Number of channels of the intermediate steps
        kernel_size (int): Size of the kernel
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
