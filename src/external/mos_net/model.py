from typing import Union

from torchaudio.transforms import Resample
import torch
from torch import nn
from wvmos import get_wvmos


class MosNet(nn.Module):

    def __init__(self, device: str = "cpu"):
        super(MosNet, self).__init__()
        self.sr = 16_000
        self.wvmos_net = get_wvmos(cuda=False)
        self.processor = self.wvmos_net.processor
        self.device = device

    def forward(self, wavs: torch.Tensor, mean: bool = True) -> Union[float, torch.Tensor]:
        x = self.processor(
            wavs, return_tensors="pt", padding=True, sampling_rate=self.sr).input_values.squeeze(0).to(self.device)
        pred_mos = self.wvmos_net(x).squeeze()
        return pred_mos if not mean else pred_mos.mean()

    def to(self, device):
        self.device = device
        return super(MosNet, self).to(device)
