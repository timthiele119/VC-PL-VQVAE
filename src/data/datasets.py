import sys
from abc import abstractmethod
import os

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import MuLawEncoding
from tqdm import tqdm
from typing import Union, List, Tuple
import yaml

from src.data.utils import list_audio_files
from src.data import preprocessing
from src.params import global_params


class VCDataset(Dataset):
    def __init__(self, root_dir: str, sr: float, receptive_field_size: int, segment_size: int,
                 dataset_specific_config: dict, normalize: bool = False):
        super(VCDataset, self).__init__()
        self.root = root_dir
        self.root_info_path = os.path.join(self.root, "info.csv")
        self.root_dataset_path = os.path.join(self.root, "dataset.npz")
        self.sr = sr
        self.receptive_field_size = receptive_field_size
        self.segment_size = segment_size
        self.step_size = self.segment_size - self.receptive_field_size
        self.dataset_specific_config = dataset_specific_config
        self.normalize = normalize
        self.mu_law_encoder = MuLawEncoding(global_params.MU_QUANTIZATION_CHANNELS)
        if not os.path.isfile(self.root_info_path):
            self._create_dataset()
        self.audio_info = self._load_audio_info()
        self.audio_data = self._load_audio_data()
        self.audio_segment_index = self._create_audio_segment_index()

    @abstractmethod
    def _get_speaker_audio_files(self) -> List[Tuple[int, str]]:
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
        quantized_audio = self.mu_law_encoder(torch.tensor(audio)).numpy().astype(getattr(np,
                                                                            global_params.MU_QUANTIZATION_NUMPY_DTYPE))
        return quantized_audio, sr, n_samples

    def _load_audio_info(self) -> dict[str, tuple[int, int, int]]:
        print("Load audio info")
        info = pd.read_csv(self.root_info_path)
        return {audio_idx: (speaker_idx, sr, n_samples + self.receptive_field_size)
                for speaker_idx, audio_idx, sr, n_samples in list(info.itertuples(index=False, name=None))}

    def _load_audio_data(self) -> dict[str, np.ndarray]:
        print("Load audio datasets")
        data = np.load(self.root_dataset_path, mmap_mode="r")
        return data

    def _create_audio_segment_index(self) -> List[Tuple[str, int]]:
        print("Create audio segment index")
        index = []
        for audio_idx in tqdm(self.audio_info.keys()):
            _, _, n_samples = self.audio_info[audio_idx]
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
        return len(self.audio_segment_index)

    def __getitem__(self, item):
        audio_idx, offset = self.audio_segment_index[item]
        zero_pads = self.mu_law_encoder(torch.zeros(self.receptive_field_size)).numpy()
        audio = np.concatenate((zero_pads, self.audio_data[audio_idx]))
        audio = audio[offset:offset+self.segment_size]
        audio = torch.tensor(audio).unsqueeze(0)
        speaker_idx = torch.tensor(self.audio_info[audio_idx][0])
        return audio, speaker_idx

    def get_raw_audios(self):
        raw_audios = []
        for audio_idx, (speaker, sr, _) in self.audio_info.items():
            raw_audios.append((self.audio_data[audio_idx], sr, speaker))
        return raw_audios


class VCTKDataset(VCDataset):
    def _get_speaker_audio_files(self) -> List[Tuple[int, str]]:
        speakers = open(self.dataset_specific_config["vctk_speaker_list"]).read().split("\n")
        relative_audio_file_paths = open(self.dataset_specific_config["vctk_relative_audio_path_list"])\
            .read().split("\n")
        vctk_audio_dir = self.dataset_specific_config["vctk_audio_dir"]
        speaker_audio_files = []
        for speaker_id, speaker in enumerate(speakers):
            speaker_dir = os.path.join(vctk_audio_dir, speaker)
            speaker_audio_files += [(speaker_id, audio_file) for audio_file in list_audio_files(speaker_dir)
                                    if "/".join(audio_file.split("/")[-2:]) in relative_audio_file_paths]
        return speaker_audio_files


class VCMelDataset(Dataset):

    def __init__(self, root_dir: str, dataset_specific_config: dict, sr: int, n_fft: int, hop_length: int,
                 win_length: int, n_mels: int):
        super(VCMelDataset, self).__init__()
        self.root = root_dir
        self.root_info_path = os.path.join(self.root, "info.csv")
        self.root_dataset_path = os.path.join(self.root, "dataset.npz")
        self.dataset_specific_config = dataset_specific_config
        self.audio_preprocessing = preprocessing.AudioPreprocessor([
            preprocessing.Normalize(),
            preprocessing.WavToMel(n_fft, hop_length, win_length, n_mels),
            preprocessing.LogShiftScaleMel()
        ])
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        if not os.path.isfile(self.root_info_path):
            self._create_dataset()
        self.audio_info = self._load_audio_info()
        self.audio_data = self._load_audio_data()

    @abstractmethod
    def _get_speaker_audio_files(self) -> List[Tuple[int, str]]:
        pass

    def _create_dataset(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        audios, info = [], {"speaker_idx": [], "audio_idx": [], "sr": [], "n_fft": [], "hop_length": [], "n_mels": []}
        for audio_idx, (speaker_idx, audio_file) in enumerate(tqdm(self._get_speaker_audio_files())):
            audio_mel = self._process_speaker_audio_file(audio_file)
            audios.append(audio_mel)
            info["speaker_idx"].append(speaker_idx)
            info["audio_idx"].append(f"arr_{audio_idx}")
            info["sr"].append(self.sr)
            info["n_fft"].append(self.n_fft)
            info["hop_length"].append(self.hop_length)
            info["n_mels"].append(self.n_mels)
        pd.DataFrame(info).to_csv(self.root_info_path, index=False)
        np.savez_compressed(self.root_dataset_path, *audios)

    def _process_speaker_audio_file(self, audio_file: str) -> np.ndarray:
        audio_wav, _ = librosa.load(path=audio_file, sr=self.sr)
        audio_mel = self.audio_preprocessing(torch.tensor(audio_wav))
        return audio_mel.numpy()

    def _load_audio_info(self) -> dict[str, tuple[int, int, int]]:
        print("Load audio info")
        info = pd.read_csv(self.root_info_path)
        return {audio_idx: (speaker_idx, sr, n_fft, hop_length, n_mels)
                for speaker_idx, audio_idx, sr, n_fft, hop_length, n_mels
                in list(info.itertuples(index=False, name=None))}

    def _load_audio_data(self) -> dict[str, np.ndarray]:
        print("Load audio datasets")
        data = np.load(self.root_dataset_path, mmap_mode="r")
        return data

    def __len__(self):
        return len(self.audio_info)

    def __getitem__(self, item):
        audio_idx = f"arr_{item}"
        audio_mel = torch.tensor(self.audio_data[audio_idx])
        speaker = self.audio_info[audio_idx][0]
        return audio_mel, speaker


class VCTKMelDataset(VCMelDataset):

    def _get_speaker_audio_files(self) -> List[Tuple[int, str]]:
        speakers = open(self.dataset_specific_config["vctk_speaker_list"]).read().split("\n")
        relative_audio_file_paths = open(self.dataset_specific_config["vctk_relative_audio_path_list"]) \
            .read().split("\n")
        vctk_audio_dir = self.dataset_specific_config["vctk_audio_dir"]
        speaker_audio_files = []
        for speaker_id, speaker in enumerate(speakers):
            speaker_dir = os.path.join(vctk_audio_dir, speaker)
            speaker_audio_files += [(speaker_id, audio_file) for audio_file in list_audio_files(speaker_dir)
                                    if "/".join(audio_file.split("/")[-2:]) in relative_audio_file_paths]
        return speaker_audio_files


class VCDatasetFactory:

    @classmethod
    def from_config(cls, config_path: str) -> Union[VCDataset, VCMelDataset]:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return cls.create_dataset(config["class_name"], config["init_args"])

    @classmethod
    def create_dataset(cls, class_name: str, init_args: dict) -> VCDataset:
        return getattr(sys.modules[__name__], class_name)(**init_args)
