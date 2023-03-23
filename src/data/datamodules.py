import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.datasets import VCDatasetFactory
from src.data.utils import VCCollateFn


class VCDataModule(pl.LightningDataModule):

    def __init__(self, train_dataset_class_name: str, train_dataset_init_args: dict, val_dataset_class_name: str,
                 val_dataset_init_args: dict, batch_size: int):
        super(VCDataModule, self).__init__()
        self.train_dataset_class_name = train_dataset_class_name
        self.train_dataset_init_args = train_dataset_init_args
        self.val_dataset_class_name = val_dataset_class_name
        self.val_dataset_init_args = val_dataset_init_args
        self.batch_size = batch_size
        self.train_dataset, self.val_dataset = None, None
        self.collate_fn = VCCollateFn()

    def prepare_data(self):
        print("Create Train Dataset")
        self.train_dataset = VCDatasetFactory.create_dataset(self.train_dataset_class_name,
                                                             self.train_dataset_init_args)
        print("Create Val Dataset")
        self.val_dataset = VCDatasetFactory.create_dataset(self.val_dataset_class_name, self.val_dataset_init_args)

    def setup(self, stage: str):
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
