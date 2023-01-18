import os
import torch
from torch.nn.utils.rnn import pad_sequence


def list_audio_files(location):
    audio_files = []
    for root, dirs, files in os.walk(location):
        for filename in [f for f in files if f.endswith((".mp3", ".wav", ".aif", "aiff", ".flac"))]:
            audio_files.append(os.path.join(root, filename))
    return audio_files


def vc_pad_collate(batch):
    waveforms = [waveform.T for waveform, _, _ in batch]
    speakers = torch.LongTensor([speaker for _, speaker, _ in batch])
    len_waveforms = torch.LongTensor([len for _, _, len in batch])
    padded_waveforms = torch.transpose(pad_sequence(waveforms, batch_first=True), 1, 2)
    return padded_waveforms, speakers, len_waveforms
