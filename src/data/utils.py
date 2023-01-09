import random
import numpy as np
import os
import torch
from torch.nn.utils.rnn import pad_sequence


def list_audio_files(location):
    audio_files = []
    for root, dirs, files in os.walk(location):
        for filename in [f for f in files if f.endswith((".mp3", ".wav", ".aif", "aiff", ".flac"))]:
            audio_files.append(os.path.join(root, filename))
    return audio_files


def mu_law_encoding(audio: np.ndarray, quantization_channels: int) -> np.ndarray:
    """
    Quantize waveform amplitudes.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)
    quantize_space = np.linspace(-1, 1, quantization_channels)
    quantized_audio = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
    quantized_audio = np.digitize(quantized_audio, quantize_space) - 1
    return quantized_audio


def mu_law_decoding(quantized_audio: np.ndarray, quantization_channels) -> np.ndarray:
    """
    Recovers waveform from quantized values.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)
    expanded = (quantized_audio / quantization_channels) * 2. - 1
    audio = np.sign(expanded) * (np.exp(np.abs(expanded) * np.log(mu + 1)) - 1) / mu
    return audio


def vc_pad_collate(batch):
    waveforms = [waveform.T for waveform, _, _ in batch]
    speakers = torch.LongTensor([speaker for _, speaker, _ in batch])
    len_waveforms = torch.LongTensor([len for _, _, len in batch])
    padded_waveforms = torch.transpose(pad_sequence(waveforms, batch_first=True), 1, 2)
    return padded_waveforms, speakers, len_waveforms