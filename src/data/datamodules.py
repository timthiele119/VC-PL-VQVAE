import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
from src.data.datasets import VCDatasetFactory
from src.params import global_params

VAL_SIZE = 10_000
TEST_SIZE = 10_000
SEED = 42


class VCDataModule(pl.LightningDataModule):

    def __init__(self, dataset_class_name: str, dataset_init_args: dict, batch_size: int = 32):
        super(VCDataModule, self).__init__()
        self.dataset_class_name = dataset_class_name
        self.dataset_init_args = dataset_init_args
        self.batch_size = batch_size
        self.dataset, self.train_dataset, self.val_dataset, self.test_dataset = None, None, None, None

    def prepare_data(self):
        self.dataset = VCDatasetFactory.create_dataset(self.dataset_class_name, self.dataset_init_args)

    def setup(self, stage: str):
        lengths = [len(self.dataset) - VAL_SIZE - TEST_SIZE, VAL_SIZE, TEST_SIZE]
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, lengths, generator=torch.Generator().manual_seed(SEED))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
