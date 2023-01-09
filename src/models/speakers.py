import torch
from torch import nn


class SpeakerEmbedding(nn.Module):

    def __init__(self, num_speakers: int, speaker_dim: int):
        super(SpeakerEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_speakers, embedding_dim=speaker_dim)

    def forward(self, speaker_indices: torch.Tensor):
        return self.embedding(speaker_indices)
