from abc import abstractmethod
import os
import librosa
import numpy as np
import pandas as pd
from src.data.utils import list_audio_files, mu_law_encoding, train_test_split
from src.params import Params, global_params
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

SEED = 42
TEST_SET_SIZE = 0.01


class VCDataset(Dataset):
    def __init__(self, params: Params, train: bool = True):
        super(VCDataset, self).__init__()
        self.root = params.ROOT_DIR
        self.root_info_path = os.path.join(self.root, "info.csv")
        self.root_dataset_path = os.path.join(self.root, "dataset.npz")
        self.sr = params.SAMPLING_RATE
        self.receptive_field_size = params.RECEPTIVE_FIELD_SIZE
        self.segment_size = params.SEGMENT_SIZE
        self.step_size = self.segment_size - self.receptive_field_size
        self.normalize = params.NORMALIZE
        self.dataset_specific_config = params.DATASET_SPECIFIC_CONFIC
        if not os.path.isfile(self.root_info_path):
            self._create_dataset()
        self.audio_info = self._load_audio_info()
        self.audio_data = self._load_audio_data()
        train_audio_ids, test_audio_ids = train_test_split(list(self.audio_info.keys()), TEST_SET_SIZE, SEED)
        self.split_audio_ids = train_audio_ids if train else test_audio_ids
        self.audio_segment_index = self._create_audio_segment_index()

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
        quantized_audio = mu_law_encoding(audio, global_params.MU_QUANTIZATION_CHANNELS)\
            .astype(getattr(np, global_params.MU_QUANTIZATION_NUMPY_DTYPE))
        return quantized_audio, sr, n_samples

    def _load_audio_info(self) -> dict[str, tuple[int, int, int]]:
        print("Load audio info")
        info = pd.read_csv(self.root_info_path)
        return {audio_idx: (speaker_idx, sr, n_samples + self.receptive_field_size)
                for speaker_idx, audio_idx, sr, n_samples in list(info.itertuples(index=False, name=None))}

    def _load_audio_data(self) -> dict[str, np.ndarray]:
        print("Load audio data")
        data = np.load(self.root_dataset_path, mmap_mode="r")
        return data

    def _create_audio_segment_index(self) -> list[tuple[str, int]]:
        print("Create audio segment index")
        index = []
        for audio_idx in tqdm(self.split_audio_ids):
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
        zero_pads = mu_law_encoding(np.zeros(self.receptive_field_size), global_params.MU_QUANTIZATION_CHANNELS)
        audio = np.concatenate((zero_pads, self.audio_data[audio_idx]))
        audio = audio[offset:offset+self.segment_size]
        audio = torch.tensor(audio).unsqueeze(0)
        speaker_idx = torch.tensor(self.audio_info[audio_idx][0])
        return audio, speaker_idx


class VCTKDataset(VCDataset):

    def __init__(self, params: Params, *args, **kwargs):
        super(VCTKDataset, self).__init__(params, *args, **kwargs)
        self.vctk_speaker_info_path = self.dataset_specific_config["VCTK_SPEAKER_INFO_PATH"]
        self.vctk_audio_dir = self.dataset_specific_config["VCTK_AUDIO_DIR"]

    def _get_speaker_audio_files(self) -> list[tuple[int, str]]:
        with open(self.vctk_speaker_info_path) as f:
            speakers = [line.split(" ")[0] for line in f.readlines()[1:]]
        speaker_audio_files = []
        for speaker_id, speaker in enumerate(speakers):
            speaker_dir = os.path.join(self.vctk_audio_dir, speaker)
            speaker_audio_files += [(speaker_id, audio_file) for audio_file in list_audio_files(speaker_dir)]
        return speaker_audio_files
