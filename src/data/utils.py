import os
from typing import List, Tuple

import torch


def list_audio_files(location: str) -> List[str]:
    audio_files = []
    for root, dirs, files in os.walk(location):
        for filename in [f for f in files if f.endswith((".mp3", ".wav", ".aif", "aiff", ".flac"))]:
            audio_files.append(os.path.join(root, filename))
    return audio_files


class MelSpecCollateFn(object):

    def __init__(self, max_seq_len: int = 1024):
        self.max_seq_len = max_seq_len

    def __call__(self, batch) -> Tuple[torch.Tensor, torch.LongTensor]:
        batch_size = len(batch)
        n_mels = batch[0][0].size(0)
        batch_seq_lengths = [seq.size(1) for seq, _ in batch]
        batch_max_seq_len = min(self.max_seq_len, max(batch_seq_lengths))
        batch_max_seq_len = batch_max_seq_len + (8 - batch_max_seq_len % 8) * (batch_max_seq_len % 8 != 0)
        padded_audio_mels = torch.zeros(batch_size, n_mels, batch_max_seq_len)
        speakers = torch.tensor([speaker for _, speaker in batch])
        for i, (audio_mel, _) in enumerate(batch):
            seq_len = batch_seq_lengths[i]
            if batch_max_seq_len < seq_len:
                start = torch.randint(low=0, high=seq_len-batch_max_seq_len, size=[1])
                padded_audio_mels[i] = audio_mel[:, start: start + batch_max_seq_len]
            else:
                n_fits = batch_max_seq_len // seq_len
                for j in range(n_fits):
                    padded_audio_mels[i, :, seq_len*j: seq_len*(j+1)] = audio_mel
                remainder = batch_max_seq_len % seq_len
                if remainder > 0:
                    padded_audio_mels[i, :, -remainder:] = audio_mel[:, :remainder]
        return padded_audio_mels, speakers
