from typing import Tuple

import torch
from torch import nn


class HleConv1d(nn.Conv1d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            dilation: int = 1,
            stride: int = 1,
            groups: int = 1
    ):
        pad = int((dilation * (kernel_size - 1) + 1 - stride) / 2)
        super(HleConv1d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        dilation=dilation, stride=stride, groups=groups, padding=pad)
        torch.nn.init.xavier_uniform_(self.weight)


class HleConvTranspose1d(nn.ConvTranspose1d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            dilation: int = 1,
            stride: int = 1,
            groups: int = 1
    ):
        pad = int((dilation * (kernel_size - 1) + 1 - stride) / 2)
        super(HleConvTranspose1d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        dilation=dilation, stride=stride, groups=groups, padding=pad)
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain("linear"))


class WaveNetLikeCell(nn.Module):

    def __init__(
            self,
            residual_dim: int,
            gate_dim: int,
            skip_dim: int,
            cond_dim: int = None,
            kernel_size: int = 1,
            dilation: int = 1
    ):
        super(WaveNetLikeCell, self).__init__()
        self.gate_dim = gate_dim
        self.in_seq_layer = nn.Sequential(
            HleConv1d(in_channels=residual_dim, out_channels=2*gate_dim, kernel_size=kernel_size, dilation=dilation),
            nn.InstanceNorm1d(2*gate_dim, momentum=0.25)
        )
        if cond_dim is not None:
            self.in_cond_layer = nn.Sequential(
                nn.Linear(in_features=cond_dim, out_features=2*gate_dim),
                nn.ReLU(inplace=True)
            )
        self.residual_layer = nn.Sequential(
            HleConv1d(in_channels=gate_dim, out_channels=residual_dim, kernel_size=kernel_size),
            nn.InstanceNorm1d(residual_dim, momentum=0.25)
        )
        self.skip_layer = nn.Sequential(
            HleConv1d(in_channels=gate_dim, out_channels=skip_dim, kernel_size=kernel_size),
            nn.InstanceNorm1d(skip_dim, momentum=0.25)
        )

    def forward(self, seq: torch.Tensor, cond: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = self.in_cond_layer(cond).unsqueeze(-1) if cond is not None and self.in_cond_layer is not None else 0
        gate = self.in_seq_layer(seq) + cond
        sigmoid_act = torch.sigmoid(gate[:, self.gate_dim:, :])
        tanh_act = torch.tanh(gate[:, :self.gate_dim, :])
        gated_seq = sigmoid_act * tanh_act
        residual = self.residual_layer(gated_seq)
        skip = self.skip_layer(gated_seq)
        return seq + residual, skip


class WaveNetLikeStack(nn.Module):

    def __init__(
            self,
            residual_dim: int,
            gate_dim: int,
            skip_dim: int,
            cond_dim: int = None,
            kernel_size: int = 1,
            dilation_steps: int = 1,
            dilation_repeats: int = 1
    ):
        super(WaveNetLikeStack, self).__init__()
        self.stack = nn.ModuleList()
        for _ in range(dilation_repeats):
            for dilation_step in range(dilation_steps):
                dilation = 2 ** dilation_step
                self.stack.append(WaveNetLikeCell(residual_dim, gate_dim, skip_dim, cond_dim, kernel_size, dilation))

    def forward(self, seq: torch.Tensor, cond: torch.Tensor = None):
        residual, skip = seq, 0
        for cell in self.stack:
            residual, _skip = cell(residual, cond)
            skip += _skip
        return skip


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
