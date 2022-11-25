from src.models.vqvae import VQVAE
import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F


def embedding_vector_usage_freq(model: VQVAE, dataloader: DataLoader, device: str = "cpu"):
    embedding_indices = []
    with torch.no_grad():
        model = model.to(device)
        for inputs in dataloader:
            inputs = inputs.to(device)
            _, indices, _ = model.quantizer(model.encoder(inputs))
            embedding_indices.append(indices)
        embedding_indices = torch.cat(embedding_indices, dim=0)
        embedding_indices = F.one_hot(embedding_indices, model.quantizer.K)
        embedding_indices = torch.sum(embedding_indices, dim=0)
    return embedding_indices

