from typing import Any, Tuple, List

import pytorch_lightning as pl
import torch
from torch import optim

from src.losses.vqvae_losses import HierarchicalVqVaeLoss
from src.modules.decoders import HleDecoder
from src.modules.encoders import HleEncoder
from src.modules.quantizers import VectorQuantizer
from src.modules.speakers import SpeakerEmbedding


class HleVqVaeVc(pl.LightningModule):

    def __init__(
            self,
            speaker_embedding: SpeakerEmbedding,
            encoder_bot: HleEncoder,
            encoder_mid: HleEncoder,
            encoder_top: HleEncoder,
            quantizer_bot: VectorQuantizer,
            quantizer_mid: VectorQuantizer,
            quantizer_top: VectorQuantizer,
            decoder_bot: HleDecoder,
            decoder_mid: HleDecoder,
            decoder_top: HleDecoder,
            learning_rate: float = 0.0005
    ):
        super(HleVqVaeVc, self).__init__()
        self.save_hyperparameters(logger=False)
        self.speaker_embedding = speaker_embedding
        self.encoder_bot, self.encoder_mid, self.encoder_top = encoder_bot, encoder_mid, encoder_top
        self.quantizer_bot, self.quantizer_mid, self.quantizer_top = quantizer_bot, quantizer_mid, quantizer_top
        self.decoder_bot, self.decoder_mid, self.decoder_top = decoder_bot, decoder_mid, decoder_top
        self.loss_fn = HierarchicalVqVaeLoss(beta=0.25, reconstruction_loss_fn="mse")
        self.learning_rate = learning_rate

    def forward(self, audio: torch.Tensor, speaker: torch.Tensor) \
            -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        u0 = audio
        s = self.speaker_embedding(speaker)
        u1, z1 = self.encoder_bot(u0)
        u2, z2 = self.encoder_mid(u1)
        _, z3 = self.encoder_top(u2)
        q1 = self.quantizer_bot(z1)
        q2 = self.quantizer_mid(z2)
        q3 = self.quantizer_top(z3)
        st_q1 = z1 + torch.detach(q1 - z1)
        st_q2 = z2 + torch.detach(q2 - z2)
        st_q3 = z3 + torch.detach(q3 - z3)
        v2 = self.decoder_top(st_q3, s)
        v1 = self.decoder_mid(torch.cat([st_q2, v2], dim=1), s)
        v0 = self.decoder_bot(torch.cat([st_q1, v1], dim=1), s)
        reconstruction, encodings, embeddings = v0, [z1, z2, z3], [q1, q2, q3]
        return reconstruction, encodings, embeddings

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        audio, mfcc, _, speaker, _ = batch
        reconstruction, embeddings, encodings = self(audio, speaker)
        loss = self.loss_fn(audio, reconstruction, embeddings, encodings)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int):
        audio, mfcc, _, speaker, _ = batch
        reconstruction, embeddings, encodings = self(audio, speaker)
        loss = self.loss_fn(audio, reconstruction, embeddings, encodings)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
