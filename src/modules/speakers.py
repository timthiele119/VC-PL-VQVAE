import torch
from torch import nn


class SpeakerEmbedding(nn.Module):
    """
    Speaker embedding, whereby the embedding matrix is initialized using Xavier initialization

    Args:
        num_speakers (int): Number of speakers
        speaker_dim (int): Dimensionality of each speaker vector
    """

    def __init__(self, num_speakers: int, speaker_dim: int):
        super(SpeakerEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_speakers, embedding_dim=speaker_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, speaker_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(speaker_indices)
