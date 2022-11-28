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
        out_quantizer = self.quantizer(encodings)
        embeddings = out_quantizer["embeddings"]
        # Straight through reparameterization trick
        st_quantized_latents = encodings + torch.detach(embeddings - encodings)
        reconstructions = self.decoder(st_quantized_latents)

        return {
            "reconstructions": reconstructions,
            "encodings": encodings,
            "embeddings": embeddings,
            "embedding_ids": out_quantizer["embedding_ids"],
            "embedding_distances": out_quantizer["embedding_distances"]
        }
