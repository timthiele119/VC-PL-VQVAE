from abc import abstractmethod
import os
import librosa
import numpy as np
import pandas as pd
from src.data.utils import list_audio_files, mu_law_encoding
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

QUANTIZATION_CHANNELS = 256
QUANTIZATION_DTYPE = np.uint8


class VCDataset(Dataset):
    def __init__(self, root: str, sr: int, receptive_field_size: int, segment_size: int, normalize: bool = False):
        super(VCDataset, self).__init__()
        self.root = root
        self.root_info_path = os.path.join(self.root, "info.csv")
        self.root_dataset_path = os.path.join(self.root, "dataset.npz")
        self.sr = sr
        self.receptive_field_size = receptive_field_size
        self.segment_size = segment_size
        self.step_size = self.segment_size - self.receptive_field_size
        self.normalize = normalize
        if not os.path.isfile(self.root_info_path):
            self._create_dataset()
        self.audio_info = self._load_audio_info()
        self.audio_data = self._load_audio_data()
        self.segment_index = self._create_segment_index()

    @abstractmethod
    def _get_speaker_audio_files(self) -> list[tuple[int, str]]:
        pass

    def _create_dataset(self):
        print("Create dataset")
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        audios, info = [], {"speaker_idx": [], "audio_idx": [], "sr": [], "n_samples": []}
        for audio_idx, (speaker_idx, audio_file) in enumerate(tqdm(self._get_speaker_audio_files())):
            audio, sr, n_samples = self._process_speaker_audio_file(audio_file)
            audios.append(audio)
            info["speaker_idx"].append(speaker_idx)
            info["audio_idx"].append(f"arr_{audio_idx}")
            info["sr"].append(sr)
            info["n_samples"].append(n_samples)
        pd.DataFrame(info).to_csv(self.root_info_path, index=False)
        np.savez_compressed(self.root_dataset_path, *audios)

    def _process_speaker_audio_file(self, audio_file):
        audio, sr = librosa.load(path=audio_file, sr=self.sr)
        n_samples = len(audio)
        audio = librosa.util.normalize(audio) if self.normalize else audio
        quantized_audio = mu_law_encoding(audio, QUANTIZATION_CHANNELS).astype(QUANTIZATION_DTYPE)
        return quantized_audio, sr, n_samples

    def _load_audio_info(self) -> dict[str, tuple[int, int, int]]:
        info = pd.read_csv(self.root_info_path)
        return {audio_idx: (speaker_idx, sr, n_samples + self.receptive_field_size)
                for speaker_idx, audio_idx, sr, n_samples in list(info.itertuples(index=False, name=None))}

    def _load_audio_data(self) -> dict[str, np.ndarray]:
        zero_pads = mu_law_encoding(np.zeros(self.receptive_field_size), QUANTIZATION_CHANNELS)
        data = {key: np.concatenate((zero_pads, audio))
                for key, audio in np.load(self.root_dataset_path, mmap_mode="r").items()}
        return data

    def _create_segment_index(self) -> list[tuple[str, int]]:
        print("Create segment index")
        index = []
        for audio_idx, (_, _, n_samples) in self.audio_info.items():
            offset = 0
            while offset + self.receptive_field_size < n_samples:
                remaining_n_samples = n_samples - offset - self.receptive_field_size
                if remaining_n_samples >= self.step_size:
                    index.append((audio_idx, offset))
                else:
                    index.append((audio_idx, n_samples - self.segment_size))
                offset += self.step_size
        return index

    def __len__(self):
        return len(self.segment_index)

    def __getitem__(self, item):
        audio_idx, offset = self.segment_index[item]
        audio = self.audio_data[audio_idx][offset:offset+self.segment_size]
        audio = torch.tensor(audio).unsqueeze(0)
        speaker_idx = torch.tensor(self.audio_info[audio_idx][0])
        return audio, speaker_idx


class VCTKDataset(VCDataset):

    def __init__(self, root: str, sr: int, receptive_field_size: int, segment_size: int,
                 vctk_speaker_info_path: str, vctk_audio_dir: str, normalization: bool = False):
        self.vctk_speaker_info_path = vctk_speaker_info_path
        self.vctk_audio_dir = vctk_audio_dir
        super(VCTKDataset, self).__init__(root, sr, receptive_field_size, segment_size, normalization)

    def _get_speaker_audio_files(self) -> list[tuple[int, str]]:
        with open(self.vctk_speaker_info_path) as f:
            speakers = [line.split(" ")[0] for line in f.readlines()[1:]]
        speaker_audio_files = []
        for speaker_id, speaker in enumerate(speakers):
            speaker_dir = os.path.join(self.vctk_audio_dir, speaker)
            speaker_audio_files += [(speaker_id, audio_file) for audio_file in list_audio_files(speaker_dir)]
        return speaker_audio_files
