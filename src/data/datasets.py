import os
from torch.utils.data import Dataset
import torchaudio


class VCTKDataset(Dataset):

    def __init__(self, speaker_info_path: str, audio_directory: str):
        with open(speaker_info_path) as f:
            speakers = [line.split(" ")[0] for line in f.readlines()[1:]]
        self.audio_speaker_info = []
        for speaker_id, speaker in enumerate(speakers):
            speaker_dir = os.path.join(audio_directory, speaker)
            for file in os.listdir(speaker_dir):
                audio_path = os.path.join(speaker_dir, file)
                self.audio_speaker_info.append((audio_path, speaker_id+1))

    def __len__(self):
        return len(self.audio_speaker_info)

    def __getitem__(self, item):
        audio_path, speaker_id = self.audio_speaker_info[item]
        waveform, sr = torchaudio.load(audio_path)
        len_waveform = waveform.shape[1]
        return waveform, speaker_id, len_waveform