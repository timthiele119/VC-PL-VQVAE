from typing import Union

from torchaudio.transforms import Resample
import torch
from torch import nn
from wvmos import get_wvmos


class MosNet(nn.Module):

    def __init__(self, sr: int = 24_000, device: str = "cpu"):
        super(MosNet, self).__init__()
        self.sr = 16_000
        self.resample = Resample(orig_freq=sr, new_freq=self.sr)
        self.wvmos_net = get_wvmos(cuda=False)
        self.processor = self.wvmos_net.processor
        self.device = device

    def forward(self, wavs: torch.Tensor, mean: bool = True) -> Union[float, torch.Tensor]:
        signals = self.resample(wavs)
        x = self.processor(
            signals, return_tensors="pt", padding=True, sampling_rate=16000).input_values.squeeze(0).to(self.device)
        pred_mos = self.wvmos_net(x).squeeze()
        return pred_mos if not mean else pred_mos.mean()

    def to(self, device):
        self.device = device
        return super(MosNet, self).to(device)
