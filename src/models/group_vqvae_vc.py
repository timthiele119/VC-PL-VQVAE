from typing import Tuple

import pytorch_lightning as pl
import torch
from torch import optim
from torchaudio.transforms import MuLawDecoding
from tqdm import tqdm

from src.losses.vqvae_losses import VqVaeLoss
from src.modules.decoders import Decoder
from src.modules.encoders import Encoder
from src.modules.quantizers import VectorQuantizer
from src.modules.speakers import SpeakerEmbedding
from src.modules.wavenet import WaveNet
from src.params import global_params


class GroupVqVaeVc(pl.LightningModule):

    def __init__(self, encoder: Encoder, vector_quantizer: VectorQuantizer, decoder: Decoder,
                 speaker_embedding: SpeakerEmbedding, wavenet: WaveNet, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.mu_law_decoder = MuLawDecoding(global_params.MU_QUANTIZATION_CHANNELS)
        self.encoder = encoder
        self.vector_quantizer = vector_quantizer
        self.decoder = decoder
        self.speaker_embedding = speaker_embedding
        self.wavenet = wavenet
        self.receptive_field_size = self.wavenet.receptive_field_size
        self.learning_rate = learning_rate

        self.loss_fn = VqVaeLoss(beta=0.25, reconstruction_loss_fn="cross_entropy")

    def forward(self, audios: torch.Tensor, speakers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_samples = audios.size(-1)
        audios = self.mu_law_decoder(audios)
        encodings = self.encoder(audios)
        embeddings = self.vector_quantizer(encodings)
        # Straight through re-parameterization trick
        st_embeddings = encodings + torch.detach(embeddings - encodings)
        decodings = self.decoder(st_embeddings)[:, :, :n_samples]
        speaker_embeddings = self.speaker_embedding(speakers)
        reconstructions = self.wavenet(audios[:, :, :n_samples-1], decodings, speaker_embeddings)
        return reconstructions, encodings, embeddings

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        audios, speakers = batch
        reconstructions, encodings, embeddings = self(audios, speakers)
        audios = audios[:, :, self.receptive_field_size:]
        reconstructions = reconstructions[:, :, self.receptive_field_size:]
        loss = self.loss_fn(audios.squeeze(), reconstructions, encodings, embeddings)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        audios, speakers = batch
        reconstructions, encodings, embeddings = self(audios, speakers)
        audios = audios[:, :, self.receptive_field_size:]
        reconstructions = reconstructions[:, :, self.receptive_field_size:]
        loss = self.loss_fn(audios.squeeze(), reconstructions, encodings, embeddings)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def convert(self, audios: torch.Tensor, target_speakers: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            n_samples = audios.size(1)
            audios = audios.unsqueeze(1)
            print("Create Local and Global Conditions")
            zero_pads = torch.zeros(audios.size(0), 1, self.receptive_field_size, device=self.device)
            audios = torch.cat([zero_pads, audios], dim=-1)
            encodings = self.encoder(audios)
            embeddings = self.vector_quantizer(encodings)
            decodings = self.decoder(embeddings)[:, :, :self.receptive_field_size+n_samples]
            target_speaker_embeddings = self.speaker_embedding(target_speakers)
            print("Generate audio")
            target_audios = zero_pads
            for sample in tqdm(range(n_samples)):
                audio_segments = target_audios[:, :, sample:sample+self.receptive_field_size]
                decoding_segments = decodings[:, :, sample:sample+self.receptive_field_size+1]
                generated_audios = self.wavenet(audio_segments, decoding_segments, target_speaker_embeddings)[:, :, -1]
                generated_audios = self.mu_law_decoder(torch.argmax(generated_audios, dim=1, keepdim=True).unsqueeze(-1))
                target_audios = torch.cat([target_audios, generated_audios], dim=-1)
            return target_audios.squeeze()[:, self.receptive_field_size:]
