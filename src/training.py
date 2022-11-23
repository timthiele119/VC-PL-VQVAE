from abc import abstractmethod
from src.models.vqvae import VQVAE
from src.losses import VQVAELoss
from torch.utils.data import DataLoader
from torch.optim import Optimizer


class VQVAETrainer:

    def __init__(self, model: VQVAE, train_dataloader: DataLoader, val_dataloader: DataLoader,
                 loss_fn: VQVAELoss, optimizer: Optimizer, device: str = "cpu"):
        self.loss_fn = loss_fn
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device

    def train(self, epochs: int):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}\n-------------------")
            self._train_epoch()
            self._evaluate_epoch()

    def _train_epoch(self):
        n_steps = len(self.train_dataloader.dataset)
        for step, (X, _) in enumerate(self.train_dataloader):
            X = X.to(self.device)
            loss = self._train_step(X)

            # Print loss every 100th training step
            if step % 100 == 0:
                current_step = step * len(X)
                print(f"Loss: {loss:>7f} [{current_step:>5d} / {n_steps:>5d}]")

    def _train_step(self, inputs):
        reconstructions, latents, quantized_latents = self.model(inputs)
        loss = self.loss_fn(inputs, reconstructions, latents, quantized_latents)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _evaluate_epoch(self):
        pass
