from abc import abstractmethod
from src.models.modules import ResBlock1d
import torch
from torch import nn


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(inputs)

    @abstractmethod
    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
        pass


class UpsamplingDecoder1d(Decoder):

    def __init__(self, upsampling_steps: int):
        super(UpsamplingDecoder1d, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=2**upsampling_steps)

    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
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

    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
        outs = self.conv_in(inputs)
        for upsampling in self.upsampling_stack:
            outs = upsampling(outs)
        outs = self.conv_out(outs)
        return outs


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
