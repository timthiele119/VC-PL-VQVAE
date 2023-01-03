from src.models.base import BaseModule
from src.models.encoders import Encoder
from src.models.quantizers import VectorQuantizer
from src.models.decoders import Decoder
import torch


class VQVAE(BaseModule):

    def __init__(self, encoder: Encoder, quantizer: VectorQuantizer, decoder: Decoder):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    @classmethod
    def from_config(cls, config):
        pass

    def forward(self, inputs: torch.Tensor) -> dict:
        encodings = self.encoder(inputs)
        embeddings = self.quantizer(encodings)
        # Straight through reparameterization trick
        st_embeddings = encodings + torch.detach(embeddings - encodings)
        reconstructions = self.decoder(st_embeddings)

        return {
            "reconstructions": reconstructions,
            "encodings": encodings,
            "embeddings": embeddings
        }
