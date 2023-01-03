import torch
from torch import nn

DILATION_KERNEL_SIZE = 2
DILATION_GROWTH_FACTOR = 2


class WaveNet(nn.Module):

    def __init__(self, in_channels: int, residual_channels: int, dilation_channels: int, skip_channels: int,
                 out_channels: int, dilation_steps: int, repeats: int, use_local_conditioning: bool = False,
                 in_channels_local_condition: int = None, use_global_conditioning: bool = False,
                 in_features_global_condition: int = None):
        super(WaveNet, self).__init__()
        self.dilation_steps, self.repeats = dilation_steps, repeats
        self.receptive_field_size = self.repeats * (2 ** (self.dilation_steps + 1))
        self.in_transform = CausalConv1D(in_channels, residual_channels, DILATION_KERNEL_SIZE, dilation=1,
                                         mask_type="A")
        self.gated_activation_stack = GatedActivationResStack(residual_channels, dilation_channels, skip_channels,
                                                              DILATION_KERNEL_SIZE, DILATION_GROWTH_FACTOR,
                                                              dilation_steps, repeats, use_local_conditioning,
                                                              in_channels_local_condition, use_global_conditioning,
                                                              in_features_global_condition)
        self.out_transform = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, out_channels, 1)
        )

    def forward(self, sequences: torch.Tensor, local_conditions: torch.Tensor = None,
                global_conditions: torch.Tensor = None):
        transformed_sequences = self.in_transform(sequences)
        summed_skip_outs = self.gated_activation_stack(transformed_sequences, local_conditions, global_conditions)
        logits = self.out_transform(summed_skip_outs)
        return logits


class GatedActivationResStack(nn.Module):

    def __init__(self, in_channels: int, dilation_channels: int, skip_channels: int, kernel_size: int,
                 dilation_growth_factor: int, dilation_steps: int, repeats: int, use_local_conditioning: bool = False,
                 in_channels_local_condition: int = None, use_global_conditioning: bool = False,
                 in_features_global_condition: int = None, ignore_first_conv: bool = True):
        super(GatedActivationResStack, self).__init__()
        self.gated_activation_blocks = nn.ModuleList()
        for stack in range(repeats):
            for dilation_step in range(dilation_steps):
                if ignore_first_conv and stack == 0 and dilation_step == 0:
                    continue
                dilation = dilation_growth_factor ** dilation_step
                use_residual = not ((stack == repeats - 1) and (dilation_step == dilation_steps - 1))
                self.gated_activation_blocks.append(
                    GatedActivationResBlock(in_channels, dilation_channels, skip_channels,
                                            kernel_size, dilation, use_residual,
                                            use_local_conditioning, in_channels_local_condition,
                                            use_global_conditioning, in_features_global_condition)
                )

    def forward(self, sequences: torch.Tensor, local_conditions: torch.Tensor = None,
                global_conditions: torch.Tensor = None):
        summed_skip_outs = None
        res_outs = sequences
        for gated_activation_block in self.gated_activation_blocks:
            res_outs, skip_outs = gated_activation_block(res_outs, local_conditions, global_conditions)
            if summed_skip_outs == None:
                summed_skip_outs = skip_outs
            else:
                summed_skip_outs += skip_outs
        return summed_skip_outs


class GatedActivationResBlock(nn.Module):

    def __init__(self, in_channels: int, dilation_channels: int, skip_channels: int, kernel_size: int,
                 dilation: int = 1, use_residual: bool = True, use_local_conditioning: bool = False,
                 in_channels_local_condition: int = None, use_global_conditioning: bool = False,
                 in_features_global_condition: int = None):
        super(GatedActivationResBlock, self).__init__()
        self.gated_activation_unit = GatedActivationUnit(in_channels, dilation_channels, kernel_size,
                                                         dilation, use_local_conditioning,
                                                         in_channels_local_condition,
                                                         use_global_conditioning,
                                                         in_features_global_condition)
        self.use_residual = use_residual
        if self.use_residual:
            self.residual_1x1conv = nn.Conv1d(dilation_channels, in_channels, 1)
        self.skip_1x1conv = nn.Conv1d(dilation_channels, skip_channels, 1)

    def forward(self, sequences: torch.Tensor, local_conditions: torch.Tensor = None,
                global_conditions: torch.Tensor = None):
        gated_out = self.gated_activation_unit(sequences, local_conditions, global_conditions)
        residual_out = self.residual_1x1conv(gated_out) + sequences if self.use_residual else None
        skip_out = self.skip_1x1conv(gated_out)
        return residual_out, skip_out


class GatedActivationUnit(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1,
                 use_local_conditioning: bool = False, in_channels_local_condition: int = None,
                 use_global_conditioning: bool = False, in_features_global_condition: int = None):
        super(GatedActivationUnit, self).__init__()
        self.use_local_conditioning = use_local_conditioning
        self.use_global_conditioning = use_global_conditioning
        self.conv_sequence = CausalConv1D(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv_sequence_gating = CausalConv1D(in_channels, out_channels, kernel_size, dilation=dilation)
        if self.use_local_conditioning:
            self.conv_local_condition = nn.Conv1d(in_channels_local_condition, out_channels, 1)
            self.conv_local_condition_gating = nn.Conv1d(in_channels_local_condition, out_channels, 1)
        if self.use_global_conditioning:
            self.linear_global_condition = nn.Linear(in_features_global_condition, out_channels)
            self.linear_global_condition_gating = nn.Linear(in_features_global_condition, out_channels)

    def forward(self, sequences: torch.Tensor, local_conditions: torch.Tensor = None,
                global_conditions: torch.Tensor = None):
        out = self._compute_ungated_output(sequences, local_conditions, global_conditions)
        gate = self._compute_gate(sequences, local_conditions, global_conditions)
        return out * gate

    def _compute_ungated_output(self, sequences: torch.Tensor, local_conditions: torch.Tensor = None,
                                global_conditions: torch.Tensor = None):
        ungated_out = self.conv_sequence(sequences)
        if self.use_local_conditioning:
            ungated_out += self.conv_local_condition(local_conditions)
        if self.use_global_conditioning:
            ungated_out += self.linear_global_condition(global_conditions).unsqueeze(-1)
        return torch.tanh(ungated_out)

    def _compute_gate(self, sequences: torch.Tensor, local_conditions: torch.Tensor = None,
                      global_conditions: torch.Tensor = None):
        gate = self.conv_sequence_gating(sequences)
        if self.use_local_conditioning:
            gate += self.conv_local_condition_gating(local_conditions)
        if self.use_global_conditioning:
            gate += self.linear_global_condition_gating(global_conditions).unsqueeze(-1)
        return torch.sigmoid(gate)


class CausalConv1D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, mask_type: str = "B"):
        super(CausalConv1D, self).__init__()
        left_pad = kernel_size + (kernel_size-1) * (dilation-1) - 1
        left_pad += 1 if mask_type == "A" else 0
        pad = (left_pad, 0)
        self.pad = nn.ConstantPad1d(pad, 0.0)
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded_inputs = self.pad(inputs)
        return self.conv1d(padded_inputs)
