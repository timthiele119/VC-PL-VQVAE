from abc import ABC, abstractmethod
from typing import List

import noisereduce as nr
import torch
from torchaudio.transforms import MelSpectrogram, MuLawEncoding


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


class WavToMuLaw(AudioPreprocessingStep):

    def __init__(self, quantization_channels: int = 256):
        self.mu_law_encoder = MuLawEncoding(quantization_channels=quantization_channels)

    def __call__(self, audio: torch.Tensor):
        return self.mu_law_encoder(audio)


class LogShiftScaleMel(AudioPreprocessingStep):

    def __init__(self):
        self.log_shift = torch.tensor(1e-5)
        self.mean, self.std = torch.tensor(-4), torch.tensor(4)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return (torch.log(self.log_shift + audio) - self.mean) / self.std


class AudioPreprocessor(AudioPreprocessingStep):

    def __init__(self, preprocessing_steps: List[AudioPreprocessingStep]):
        self.preprocessing_steps = preprocessing_steps

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        for preprocessing_step in self.preprocessing_steps:
            audio = preprocessing_step(audio)
        return audio
