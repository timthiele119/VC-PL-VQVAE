from abc import abstractmethod

import torch

from src.models.vqvae import VQVAE
from src.losses.vqvae_losses import VQVAELoss
from src.training.codebook_collapse_handlers import CodebookCollapseHandler
from torch.utils.data import DataLoader
from torch.optim import Optimizer


class VQVAETrainer:

    def __init__(self, model: VQVAE, train_dataloader: DataLoader, test_dataloader: DataLoader,
                 loss_fn: VQVAELoss, optimizer: Optimizer, codebook_collapse_handler: CodebookCollapseHandler = None,
                 device: str = "cpu"):
        self.loss_fn = loss_fn
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.device = device
        self.codebook_collapse_handler = codebook_collapse_handler

    def train(self, epochs: int):
        for epoch in range(epochs):
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

    def _train_step(self, inputs: torch.Tensor) -> float:
        inputs = inputs.to(self.device)
        out = self.model(inputs)
        reconstructions, encodings, embeddings = out["reconstructions"], out["encodings"], out["embeddings"]
        loss = self.loss_fn(inputs, reconstructions, encodings, embeddings)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.codebook_collapse_handler is not None:
            self.codebook_collapse_handler.step(out)
        return loss.item()

    def _test_epoch(self):
        avg_loss = 0
        for step, inputs in enumerate(self.test_dataloader):
            inputs = inputs.to(self.device)
            loss = self._test_step(inputs)
            avg_loss += 1/(step+1) * (loss - avg_loss)
        print(f"Test Avg. Loss: {avg_loss:>5f}\n")

    def _test_step(self, inputs: torch.Tensor) -> float:
        with torch.no_grad():
            inputs = inputs.to(self.device)
            out = self.model(inputs)
            reconstructions, encodings, embeddings = out["reconstructions"], out["encodings"], out["embeddings"]
            loss = self.loss_fn(inputs, reconstructions, encodings, embeddings)
            return loss.item()