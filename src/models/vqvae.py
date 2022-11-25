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

    def forward(self, inputs: torch.Tensor) -> tuple:
        latents = self.encoder(inputs)
        quantized_latents, _, _ = self.quantizer(latents)
        # Straight through reparameterization trick
        st_quantized_latents = latents + torch.detach(quantized_latents - latents)
        reconstruction = self.decoder(st_quantized_latents)

        return reconstruction, latents, quantized_latents
