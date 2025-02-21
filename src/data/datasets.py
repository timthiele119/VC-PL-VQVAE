from collections import defaultdict
from abc import abstractmethod
import sys
import os

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict
import yaml

from src.data.utils import list_audio_files, EMOTIONS
from src.data import preprocessing


class VCDataset(Dataset):

    def __init__(self, root_dir: str, dataset_specific_config: dict, sr: int, n_fft: int, hop_length: int,
                 win_length: int, n_mels: int, n_mfcc: int = 13, features: List[str] = None):
        super(VCDataset, self).__init__()
        self.root = root_dir
        self.dataset_path = os.path.join(self.root, "dataset.npz")
        self.dataset_specific_config = dataset_specific_config
        self.wav_audio_preprocessing = preprocessing.AudioPreprocessor([
            preprocessing.Normalize()
        ])
        self.mel_audio_preprocessing = preprocessing.AudioPreprocessor([
            preprocessing.Normalize(),
            preprocessing.WavToMel(sr, n_fft, hop_length, win_length, n_mels),
            preprocessing.LogShiftScaleMel()
        ])
        self.mfcc_audio_preprocessing = preprocessing.AudioPreprocessor([
            preprocessing.Normalize(),
            preprocessing.WavToMFCC(sr, n_mfcc, n_fft, hop_length, win_length, n_mels)
        ])
        self.f0_audio_preprocessing = preprocessing.AudioPreprocessor([
            preprocessing.MelToF0()
        ])
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        if not os.path.isfile(self.dataset_path):
            self._create_dataset()
        self.data = None
        self.feature_temporal = {
            "mels": True,
            "mfccs": True,
            "wavs": True,
            "f0s": True,
            "speakers": False,
            "emotions": False,
        }
        self.feature_transpose = {
            "mels": True,
            "mfccs": True,
            "wavs": False,
            "f0s": False,
            "speakers": False,
            "emotions": False
        }
        self.feature_names = features if features is not None else list(self.feature_temporal.keys())
        self._load_dataset()

    @abstractmethod
    def _get_data(self) -> List[Dict]:
        pass

    def _create_dataset(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        speakers, emotions, mels, wavs, mfccs, f0s = [], [], [], [], [], []
        for data in self._get_data():
            speaker, emotion, audio = data["speaker"], data["emotion"], data["audio"]
            mel, wav, mfcc = self._extract_mel(audio), self._extract_wav(audio), self._extract_mfcc(audio)
            f0 = self._extract_f0(mel)
            speakers.append(speaker)
            emotions.append(emotion)
            mels.append(mel)
            wavs.append(wav)
            mfccs.append(mfcc)
            f0s.append(f0)
        np.savez_compressed(self.dataset_path, speakers=speakers, emotions=emotions, mels=mels, wavs=wavs, mfccs=mfccs,
                            f0s=f0s)

    def _extract_mel(self, audio: np.ndarray) -> np.ndarray:
        mel = self.mel_audio_preprocessing(torch.tensor(audio))
        return mel.numpy().T

    def _extract_wav(self,  audio: np.ndarray) -> np.ndarray:
        wav = self.wav_audio_preprocessing(torch.tensor(audio))
        return wav.numpy()

    def _extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        mfcc = self.mfcc_audio_preprocessing(torch.tensor(audio))
        return mfcc.numpy().T

    def _extract_f0(self, mel: torch.Tensor) -> torch.Tensor:
        f0 = self.f0_audio_preprocessing(torch.tensor(mel.T))
        return f0.numpy()

    def _load_dataset(self):
        data = np.load(self.dataset_path, allow_pickle=True)
        self.data = []
        for example_feature_values in zip(*(data[feature_name] for feature_name in self.feature_names)):
            example = {feature_name: torch.tensor(feature_value) if not self.feature_transpose[feature_name]
                                     else torch.tensor(feature_value).T
                       for feature_name, feature_value
                       in zip(self.feature_names, example_feature_values)}
            self.data.append(example)

    def get_max_speaker_id(self):
        return max([example["speakers"] for example in self.data])

    def increment_speakers(self, increment: int):
        for i in range(len(self)):
            self.data[i]["speakers"] += increment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.feature_temporal


class VCTKDataset(VCDataset):

    def _get_data(self) -> List[Dict]:
        speakers = open(self.dataset_specific_config["vctk_speaker_list"]).read().split("\n")
        relative_audio_file_paths = open(self.dataset_specific_config["vctk_relative_audio_path_list"]) \
            .read().split("\n")
        vctk_audio_dir = self.dataset_specific_config["vctk_audio_dir"]
        for speaker, speaker_str in enumerate(speakers):
            speaker_dir = os.path.join(vctk_audio_dir, speaker_str)
            audio_file_paths = [audio_file for audio_file in list_audio_files(speaker_dir)
                                if "/".join(audio_file.split("/")[-2:]) in relative_audio_file_paths]
            for audio_file_path in audio_file_paths:
                audio, _ = librosa.load(path=audio_file_path, sr=self.sr)
                emotion = EMOTIONS.index("neutral")
                yield {"speaker": speaker, "emotion": emotion, "audio": audio}


class ESDDataset(VCDataset):

    def _get_data(self) -> List[Dict]:
        esd_speakers = open(self.dataset_specific_config["esd_speaker_list"]).read().split("\n")
        esd_audio_dir = self.dataset_specific_config["esd_audio_dir"]
        esd_split = self.dataset_specific_config["esd_split"]
        for speaker, speaker_str in enumerate(esd_speakers):
            speaker_dir = os.path.join(esd_audio_dir, speaker_str)
            for emotion, emotion_dir in [(f.name, f.path) for f in os.scandir(speaker_dir) if f.is_dir()]:
                emotion = EMOTIONS.index(emotion.lower())
                audio_dir = os.path.join(emotion_dir, esd_split)
                audio_file_paths = list_audio_files(audio_dir)
                for audio_file_path in audio_file_paths:
                    audio, _ = librosa.load(path=audio_file_path, sr=self.sr)
                    yield {"speaker": speaker, "emotion": emotion, "audio": audio}


class VCDatasetFactory:

    @classmethod
    def from_config(cls, config_path: str) -> VCDataset:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return cls.create_dataset(config["class_name"], config["init_args"])

    @classmethod
    def create_dataset(cls, class_name: str, init_args: dict) -> VCDataset:
        return getattr(sys.modules[__name__], class_name)(**init_args)
