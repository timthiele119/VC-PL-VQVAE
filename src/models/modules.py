from torch import nn


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
