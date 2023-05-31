from collections import namedtuple
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

    def __call__(self, batch) -> Tuple:
        feature_temporal = [feature_temporal for _, feature_temporal in batch][0]
        batch = [example for example, _ in batch]
        feature_names = list(batch[0].keys())
        Batch = namedtuple("Batch", " ".join(feature_names))
        values = []
        for feature_name in feature_names:
            feature_values = [example[feature_name] for example in batch]
            if feature_temporal[feature_name]:
                values.append(self._pad(feature_values))
            else:
                values.append(torch.tensor(feature_values))
        return Batch(*tuple(values))

    @staticmethod
    def _pad(sequences: List[torch.Tensor], max_seq_len: int = sys.maxsize) -> torch.Tensor:
        batch_size = len(sequences)
        dim = sequences[0].dim()
        sequences = sequences if dim > 1 else [seq.unsqueeze(0) for seq in sequences]
        n_channels = sequences[0].size(0)
        batch_seq_lengths = [mel.size(1) for mel in sequences]
        batch_max_seq_len = min(max_seq_len, max(batch_seq_lengths))
        batch_max_seq_len = batch_max_seq_len + (8 - batch_max_seq_len % 8) * (batch_max_seq_len % 8 != 0)
        padded_mels = torch.zeros(batch_size, n_channels, batch_max_seq_len)
        for i, mel in enumerate(sequences):
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
        return padded_mels.squeeze()

