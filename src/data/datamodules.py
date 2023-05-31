from typing import Dict, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

from src.data.datasets import VCDatasetFactory
from src.data.utils import VCCollateFn


class VCDataModule(pl.LightningDataModule):

    def __init__(self, train_datasets: Dict[str, Dict], val_datasets: Dict[str, Dict], batch_size: int):
        super(VCDataModule, self).__init__()
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset = None, None
        self.collate_fn = VCCollateFn()

    def prepare_data(self):
        print("Create Val Dataset")
        self.val_dataset = self.prepare_datasets(self.val_datasets)

        print("Create Train Dataset")
        self.train_dataset = self.prepare_datasets(self.train_datasets)

    @staticmethod
    def prepare_datasets(datasets: Dict[str, Dict]):
        concat_datasets = []
        speaker_increment = 0
        for class_name, init_args in datasets.items():
            dataset = VCDatasetFactory.create_dataset(class_name, init_args)
            max_speaker_id = dataset.get_max_speaker_id()
            dataset.increment_speakers(speaker_increment)
            speaker_increment += max_speaker_id + 1
            concat_datasets.append(dataset)
        return ConcatDataset(concat_datasets)

    def setup(self, stage: str):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
