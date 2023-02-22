from abc import abstractmethod
from typing import Any

from src.models.modules import ResBlock1d, HleConv1d, HleConvTranspose1d, WaveNetLikeStack
import torch
from torch import nn


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        pass


class UpsamplingDecoder1d(Decoder):

    def __init__(self, upsampling_steps: int):
        super(UpsamplingDecoder1d, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=2**upsampling_steps)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.upsampler(inputs)


class LearnedUpsamplingDecoder1d(Decoder):

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, kernel_size: int,
                 upsampling_steps: int):
        super(LearnedUpsamplingDecoder1d, self).__init__()
        self.conv_in = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding="same")
        self.upsampling_stack = nn.ModuleList([
            LearnedUpsamplingResBlock1d(hidden_channels, hidden_channels, kernel_size)
            for _ in range(upsampling_steps)
        ])
        self.conv_out = nn.Conv1d(hidden_channels, out_channels, kernel_size, padding="same")

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        outs = self.conv_in(seq)
        for upsampling in self.upsampling_stack:
            outs = upsampling(outs)
        outs = self.conv_out(outs)
        return outs


class HleDecoder(Decoder):
    def __init__(
            self,
            input_dim: int,
            cond_dim: int,
            output_dim: int,
            gate_dim: int,
            residual_dim: int,
            skip_dim: int,
            kernel_size: int,
            dilation_steps: int,
            dilation_repeats: int = 1
    ):
        super(HleDecoder, self).__init__()
        self.input_layer = nn.Sequential(
            HleConv1d(in_channels=input_dim, out_channels=2*residual_dim, kernel_size=kernel_size),
            nn.InstanceNorm1d(2*residual_dim, momentum=0.25),
            nn.GLU(dim=1)
        )
        self.upsampling_layer = nn.Sequential(
            HleConvTranspose1d(in_channels=residual_dim, out_channels=2*residual_dim, kernel_size=8, stride=2),
            nn.InstanceNorm1d(2*residual_dim, momentum=0.25),
            nn.GLU(dim=1)
        )
        self.dilation_stack = WaveNetLikeStack(residual_dim=residual_dim, gate_dim=gate_dim, skip_dim=skip_dim,
                                               cond_dim=cond_dim, kernel_size=kernel_size,
                                               dilation_steps=dilation_steps, dilation_repeats=dilation_repeats)
        self.output_layer = nn.Sequential(
            HleConv1d(in_channels=residual_dim, out_channels=2 * output_dim, kernel_size=kernel_size),
            nn.InstanceNorm1d(2 * output_dim, momentum=0.25),
            nn.GLU(dim=1),
            HleConv1d(in_channels=output_dim, out_channels=output_dim, kernel_size=15)
        )


    def forward(self, seq: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        seq = self.input_layer(seq)
        seq = self.upsampling_layer(seq)
        seq = self.dilation_stack(seq, cond)
        return self.output_layer(seq)


class LearnedUpsamplingResBlock1d(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super(LearnedUpsamplingResBlock1d, self).__init__()
        self.res_block1 = ResBlock1d(in_channels, hidden_channels, kernel_size)
        self.upsampling = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        self.res_block2 = ResBlock1d(in_channels, hidden_channels, kernel_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outs = self.res_block1(inputs)
        outs = self.upsampling(outs)
        outs = self.res_block2(outs)
        return outs