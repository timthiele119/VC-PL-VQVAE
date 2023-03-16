from typing import Any, Tuple, List

import pytorch_lightning as pl
import torch
from torch import optim

from src.losses.vqvae_losses import VqVaeLoss
from src.modules.decoders import HleDecoder
from src.modules.encoders import HleEncoder
from src.modules.quantizers import EMAVectorQuantizer
from src.modules.speakers import SpeakerEmbedding


class VqVaeVc(pl.LightningModule):

    def __init__(
            self,
            speaker_embedding: SpeakerEmbedding,
            encoder: HleEncoder,
            quantizer: EMAVectorQuantizer,
            decoder: HleDecoder,
            learning_rate: float = 0.0005
    ):
        super(VqVaeVc, self).__init__()
        self.save_hyperparameters(logger=False)
        self.speaker_embedding = speaker_embedding
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.loss_fn = VqVaeLoss(beta=0.25, reconstruction_loss_fn="mse", use_codebook_loss=False)
        self.learning_rate = learning_rate

    def forward(self, audio: torch.Tensor, speaker: torch.Tensor) \
            -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        u = audio
        s = self.speaker_embedding(speaker)
        _, z = self.encoder(u)
        q = self.quantizer(z)
        st_q = z + torch.detach(q - z)
        v = self.decoder(q, st_q)
        reconstruction, encoding, embedding = v, z, q
        return reconstruction, encoding, embedding

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        audio, speaker = batch
        reconstruction, embedding, encoding = self(audio, speaker)
        loss = self.loss_fn(audio, reconstruction, embedding, encoding)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        audio, speaker = batch
        reconstruction, embedding, encoding = self(audio, speaker)
        loss = self.loss_fn(audio, reconstruction, embedding, encoding)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.75, verbose=True)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
