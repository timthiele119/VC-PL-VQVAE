from abc import ABC, abstractmethod
from typing import Union

from librosa import resample
import torch
from wvmos import get_wvmos


class Metric(ABC):

    @abstractmethod
    def __call__(self, wavs: torch.Tensor) -> Union[float, torch.Tensor]:
        pass


class MOS(Metric):

    def __init__(self, sr: int = 24_000, device: str = "cuda"):
        self.cuda = True if device == "cuda" else False
        self.sr = sr
        self.wvmos_net = get_wvmos(cuda=self.cuda)

    def __call__(self, wavs: torch.Tensor, mean: bool = True) -> Union[float, torch.Tensor]:
        pred_mos = []
        signals = [resample(y=wav.numpy(), orig_sr=self.sr, target_sr=16_000) for wav in wavs]
        for signal in signals:
            x = self.wvmos_net.processor(signal, return_tensors="pt", padding=True, sampling_rate=16000).input_values
            with torch.no_grad():
                if self.cuda:
                    x = x.cuda()
                res = self.wvmos_net.forward(x).mean()
            pred_mos.append(res.item())
        pred_mos = torch.Tensor(pred_mos)
        return pred_mos.mean() if mean else pred_mos
