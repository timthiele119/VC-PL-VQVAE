from typing import List

import torch
from torch import nn


class VqVaeLoss(nn.Module):

    def __init__(self, beta: float = 0.25, reconstruction_loss_fn: str = "cross_entropy"):
        super(VqVaeLoss, self).__init__()
        self.beta = beta
        if reconstruction_loss_fn == "cross_entropy":
            self.reconstruction_loss_fn = nn.CrossEntropyLoss()
        elif reconstruction_loss_fn == "mse":
            self.reconstruction_loss_fn = nn.MSELoss()
        self.embedding_loss_fn = nn.MSELoss()
        self.commitment_loss_fn = nn.MSELoss()

    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor, encodings: torch.Tensor,
                embeddings: torch.Tensor) -> float:
        reconstruction_loss = self.reconstruction_loss_fn(reconstructions, inputs)
        embedding_loss = self.embedding_loss_fn(embeddings, encodings.detach())
        commitment_loss = self.commitment_loss_fn(encodings, embeddings.detach())
        return reconstruction_loss + embedding_loss + self.beta * commitment_loss


class HierarchicalVqVaeLoss(nn.Module):

    def __init__(self, beta: float = 0.25, reconstruction_loss_fn: str = "mse"):
        super(HierarchicalVqVaeLoss, self).__init__()
        self.beta = beta
        if reconstruction_loss_fn == "mse":
            self.reconstruction_loss_fn = nn.MSELoss()
        elif reconstruction_loss_fn == "cross_entropy":
            self.reconstruction_loss_fn = nn.CrossEntropyLoss()
        else:
            raise Exception(f"Reconstruction loss function \"{reconstruction_loss_fn}\" not known")
        self.embedding_loss_fn = nn.MSELoss()
        self.commitment_loss_fn = nn.MSELoss()

    def forward(self, original: torch.Tensor, reconstruction: torch.Tensor, encodings: List[torch.Tensor],
                embeddings: List[torch.Tensor]):
        reconstruction_loss = self.reconstruction_loss_fn(reconstruction, original)
        embedding_loss, commitment_loss = 0, 0
        for embedding, encoding in zip(embeddings, encodings):
            embedding_loss += self.embedding_loss_fn(embedding, encoding.detach())
            commitment_loss += self.commitment_loss_fn(encoding, embedding.detach())
        return reconstruction_loss + embedding_loss + self.beta * commitment_loss
