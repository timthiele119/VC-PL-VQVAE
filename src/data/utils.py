import os
import sys
from typing import List, Tuple

import torch


EMOTIONS = ["neutral", "angry", "happy", "sad", "surprise"]


def list_audio_files(location: str) -> List[str]:
    audio_files = []
    for root, dirs, files in os.walk(location):
        for filename in [f for f in files if f.endswith((".mp3", ".wav", ".aif", "aiff", ".flac"))]:
            audio_files.append(os.path.join(root, filename))
    return audio_files


class VCCollateFn(object):

    def __init__(self, max_seq_len: int = 1024):
        self.max_seq_len = max_seq_len

    def __call__(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor]:
        mels = self._pad([mel for mel, _, _, _, _, _ in batch], max_seq_len=self.max_seq_len)
        mfccs = self._pad([mfcc for _, mfcc, _, _, _, _ in batch], max_seq_len=self.max_seq_len)
        wavs = self._pad([wav.unsqueeze(0) for _, _, wav, _, _, _ in batch]).squeeze().type(torch.int32)
        f0s = self._pad([f0.unsqueeze(0) for _, _, _, f0, _, _ in batch], max_seq_len=self.max_seq_len).squeeze()
        speakers = torch.tensor([speaker for _, _, _, _, speaker, _ in batch])
        emotions = torch.tensor([emotion for _, _, _, _, _, emotion in batch])
        return mels, mfccs, wavs, f0s, speakers, emotions

    @staticmethod
    def _pad(mels: List[torch.Tensor], max_seq_len: int = sys.maxsize) -> torch.Tensor:
        batch_size = len(mels)
        n_channels = mels[0].size(0)
        batch_seq_lengths = [mel.size(1) for mel in mels]
        batch_max_seq_len = min(max_seq_len, max(batch_seq_lengths))
        batch_max_seq_len = batch_max_seq_len + (8 - batch_max_seq_len % 8) * (batch_max_seq_len % 8 != 0)
        padded_mels = torch.zeros(batch_size, n_channels, batch_max_seq_len)
        for i, mel in enumerate(mels):
            seq_len = batch_seq_lengths[i]
            if batch_max_seq_len < seq_len:
                start = torch.randint(low=0, high=seq_len-batch_max_seq_len, size=[1])
                padded_mels[i] = mel[:, start: start + batch_max_seq_len]
            else:
                n_fits = batch_max_seq_len // seq_len
                for j in range(n_fits):
                    padded_mels[i, :, seq_len*j: seq_len*(j+1)] = mel
                remainder = batch_max_seq_len % seq_len
                if remainder > 0:
                    padded_mels[i, :, -remainder:] = mel[:, :remainder]
        return padded_mels

