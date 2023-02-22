from abc import ABC, abstractmethod
from typing import List

import noisereduce as nr
import torch
from torchaudio.transforms import MelSpectrogram


class AudioPreprocessingStep(ABC):

    @abstractmethod
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        pass


class NoiseReduction(AudioPreprocessingStep):

    def __init__(self, sr: int):
        self.sr = sr

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.numpy()
        audio = nr.reduce_noise(audio, sr=self.sr)
        return torch.from_numpy(audio)


class Normalize(AudioPreprocessingStep):

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return audio / torch.max(torch.abs(audio))


class WavToMel(AudioPreprocessingStep):

    def __init__(self, n_fft: int, hop_length: int, win_length: int, n_mels: int):
        self.to_mel = MelSpectrogram(n_mels=n_mels, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return self.to_mel(audio)


class LogShiftScaleMel(AudioPreprocessingStep):

    def __init__(self):
        self.log_shift = torch.tensor(1e-5)
        self.mean, self.std = torch.tensor(-4), torch.tensor(4)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return (torch.log(self.log_shift + audio) - self.mean) / self.std


class MelToMfcc(AudioPreprocessingStep):

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel_ndim = mel.shape[1]
        mel = mel.transpose(1, 2).unsqueeze(-1)
        mel = torch.cat([mel, torch.zeros_like(mel)], dim=-1)
        mcc = torch.fft.irfft(mel, dim=1, n=2*(mel_ndim-1)).transpose(1, 2)[:, :mel_ndim]
        mcc[:, 0] /= 2.
        return mcc.squeeze()


class MfccToMel(AudioPreprocessingStep):

    def __call__(self, mcc: torch.Tensor) -> torch.Tensor:
        if len(mcc.shape) == 2:
            mcc = mcc.unsqueeze(0)
        mcc = mcc.transpose(1, 2)
        mcc = torch.cat([mcc, torch.flip(mcc[:, :, 1:-1], dims=[-1])], dim=-1)
        mcc[:, :, 0] = mcc[:, :, 0] * 2.0
        mel = torch.fft.rfft(mcc, dim=1)
        mel = mel[:, :, :, 0]
        return mel.transpose(1, 2).squeeze()


class AudioPreprocessor(AudioPreprocessingStep):

    def __init__(self, preprocessing_steps: List[AudioPreprocessingStep]):
        self.preprocessing_steps = preprocessing_steps

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for preprocessing_step in self.preprocessing_steps:
            audio = preprocessing_step(audio)
        return audio
