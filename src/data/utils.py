import torch
from torch.nn.utils.rnn import pad_sequence


def vc_pad_collate(batch):
    waveforms = [waveform.T for waveform, _, _ in batch]
    speakers = torch.LongTensor([speaker for _, speaker, _ in batch])
    len_waveforms = torch.LongTensor([len for _, _, len in batch])
    padded_waveforms = torch.transpose(pad_sequence(waveforms, batch_first=True), 1, 2)
    return padded_waveforms, speakers, len_waveforms