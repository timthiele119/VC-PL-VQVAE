from abc import ABC, abstractmethod
from typing import List

import librosa
import noisereduce as nr
import torch
from torchaudio.transforms import MelSpectrogram, MuLawEncoding, MFCC

from src.external.jdc.model import JDCNet
from src.params import global_params


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

    def __init__(self, sr: int, n_fft: int, hop_length: int, win_length: int, n_mels: int):
        self.mel = MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                                  n_mels=n_mels)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        return self.mel(audio)


class WavToMFCC(AudioPreprocessingStep):

    def __init__(self, sr: int, n_mfcc: int, n_fft: int, hop_length: int, win_length: int, n_mels: int):
        self.mfcc = MFCC(sample_rate=sr, n_mfcc=n_mfcc,
                         melkwargs={"n_fft": n_fft, "hop_length": hop_length, "n_mels": n_mels})

    def __call__(self, audio: torch.Tensor):
        return self.mfcc(audio)


class MelToF0(AudioPreprocessingStep):

    def __init__(self):
        self.f0_model = JDCNet(num_class=1)
        self.f0_model.load_state_dict(torch.load(global_params.PATH_JDC_PARMS)["net"])

    def __call__(self, audio: torch.Tensor):
        with torch.no_grad():
            f0, _, _ = self.f0_model(audio.unsqueeze(0).unsqueeze(1))
            return f0


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
