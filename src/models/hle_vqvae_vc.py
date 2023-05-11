from typing import Any, Tuple, List

from parallel_wavegan.utils import load_model
import pytorch_lightning as pl
import torch
from torch import optim

from src.losses.vqvae_losses import HierarchicalVqVaeLoss
from src.modules.decoders import HleDecoder
from src.modules.encoders import HleEncoder
from src.external.mos_net.model import MosNet
from src.modules.quantizers import VectorQuantizer
from src.modules.speakers import SpeakerEmbedding
from src.params import global_params


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
            use_mfcc_input: bool = False,
            learning_rate: float = 0.0005,
    ):
        super(HleVqVaeVc, self).__init__()
        self.save_hyperparameters(logger=False)
        self.speaker_embedding = speaker_embedding
        self.encoder_bot, self.encoder_mid, self.encoder_top = encoder_bot, encoder_mid, encoder_top
        self.quantizer_bot, self.quantizer_mid, self.quantizer_top = quantizer_bot, quantizer_mid, quantizer_top
        self.decoder_bot, self.decoder_mid, self.decoder_top = decoder_bot, decoder_mid, decoder_top
        self.loss_fn = HierarchicalVqVaeLoss(beta=0.25, reconstruction_loss_fn="mse")
        self.use_mfcc_input = use_mfcc_input
        self.learning_rate = learning_rate
        self.vocoder = load_model(global_params.PATH_HIFIGAN_PARAMS).requires_grad_(False)
        self.mos_net = MosNet(device=self.device).requires_grad_(False)

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

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                      batch_idx: int):
        mel, mfcc, _, f0, speaker, _ = batch
        audio = mfcc if self.use_mfcc_input else mel
        reconstruction, embeddings, encodings = self(audio, speaker)
        loss = self.loss_fn(mel, reconstruction, embeddings, encodings)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                        batch_idx: int):
        mel, mfcc, _, f0, speaker, _ = batch
        audio = mfcc if self.use_mfcc_input else mel
        reconstructed_mel, embeddings, encodings = self(audio, speaker)
        loss = self.loss_fn(mel, reconstructed_mel, embeddings, encodings)
        self.log("val_loss", loss)
        reconstructed_wav = self.vocoder(reconstructed_mel).squeeze()
        self.mos_net = self.mos_net.to(self.device)
        mos = self.mos_net(reconstructed_wav)
        self.log("mos_score", mos)
        return loss

    def configure_optimizers(self) -> Any:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
