import pytorch_lightning as pl
import torch
from torch import optim
from torchaudio.transforms import MuLawDecoding
from src.losses.vqvae_losses import VQVAELoss
from src.models.decoders import Decoder
from src.models.encoders import Encoder
from src.models.quantizers import VectorQuantizer
from src.models.speakers import SpeakerEmbedding
from src.models.wavenet import WaveNet
from src.params import global_params


class VQVAEVC(pl.LightningModule):

    def __init__(self, encoder: Encoder, vector_quantizer: VectorQuantizer, decoder: Decoder,
                 speaker_embedding: SpeakerEmbedding, wavenet: WaveNet):
        super().__init__()
        self.mu_law_decoder = MuLawDecoding(global_params.MU_QUANTIZATION_CHANNELS)
        self.encoder = encoder
        self.vector_quantizer = vector_quantizer
        self.decoder = decoder
        self.speaker_embedding = speaker_embedding
        self.wavenet = wavenet
        self.receptive_field_size = self.wavenet.receptive_field_size

        self.loss_fn = VQVAELoss(beta=0.25, reconstruction_loss_fn="cross_entropy")

    def forward(self, audios: torch.Tensor, speakers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def training_step(self, batch, batch_idx):
        audios, speakers = batch
        reconstructions, encodings, embeddings = self(audios, speakers)
        audios = audios[:, :, self.receptive_field_size:]
        reconstructions = reconstructions[:, :, self.receptive_field_size:]
        loss = self.loss_fn(audios.squeeze(), reconstructions, encodings, embeddings)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        audios, speakers = batch
        reconstructions, encodings, embeddings = self(audios, speakers)
        audios = audios[:, :, self.receptive_field_size:]
        reconstructions = reconstructions[:, :, self.receptive_field_size:]
        loss = self.loss_fn(audios.squeeze(), reconstructions, encodings, embeddings)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters())
