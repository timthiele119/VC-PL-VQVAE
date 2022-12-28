import torch
from torch import nn
from src.models.decoders import LearnedUpsamplingDecoder1d
from src.models.encoders import LearnedDownsamplingEncoder1d
from src.models.quantizers import VanillaVectorQuantizer

KERNEL_SIZE = 3


class VQVAEVC(nn.Module):

    def __init__(self, hidden_channels: int, downsampling_steps: int, embedding_dim: int, num_embeddings: int,
                 speaker_dim: int, num_speakers: int):
        super(VQVAEVC, self).__init__()
        self.encoder = LearnedDownsamplingEncoder1d(1, hidden_channels, embedding_dim, KERNEL_SIZE, downsampling_steps)
        self.quantizer = VanillaVectorQuantizer(embedding_dim, num_embeddings)
        self.decoder = LearnedUpsamplingDecoder1d(embedding_dim, hidden_channels, hidden_channels, KERNEL_SIZE,
                                                  downsampling_steps)
        self.speaker_embeddings = nn.Embedding(num_speakers, speaker_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encodings = self.encoder(inputs)
        quantizer_out = self.quantizer(encodings)
        embeddings = quantizer_out["embeddings"]
        # Straight through reparameterization trick
        st_embeddings = encodings + torch.detach(embeddings - encodings)
        decodings = self.decoder(st_embeddings)
        return decodings
