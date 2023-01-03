from src.data.datasets import VCDataset
from src.losses.vqvae_losses import VQVAELoss
from src.models.vqvae_vc import VQVAEVC
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer


class VQVAEVCTrainer:

    def __init__(self, model: VQVAEVC, train_dataset: VCDataset, test_dataset: VCDataset, batch_size: int,
                 loss_fn: VQVAELoss, optimizer: Optimizer, epochs: int, device: str = "cpu"):
        self.model = model.to(device)
        self.receptive_field_size = int(self.model.wavenet.receptive_field_size)
        self.train_dataloader = DataLoader(train_dataset, batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device

    def train(self):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}\n-------------------")
            self._train_epoch()
            self._test_epoch()

    def _train_epoch(self):
        n_steps = len(self.train_dataloader.dataset)
        for step, inputs in enumerate(self.train_dataloader):
            loss = self._train_step(inputs)

            # Print loss every 100th training step
            if step % 100 == 0:
                current_step = step * len(inputs)
                print(f"Train Loss: {loss:>5f} [{current_step:>5d} / {n_steps:>5d}]")

    def _train_step(self, inputs: tuple[torch.Tensor]):
        audios, speakers = inputs[0].to(self.device), inputs[1].to(self.device)
        outputs = self.model(audios, speakers)
        reconstructions, encodings, embeddings = outputs["reconstructions"], outputs["encodings"], outputs["embeddings"]
        audios = audios[:, :, self.receptive_field_size:]
        reconstructions = reconstructions[:, :, self.receptive_field_size:]
        loss = self.loss_fn(audios.squeeze(), reconstructions, embeddings, encodings)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _test_epoch(self):
        pass
