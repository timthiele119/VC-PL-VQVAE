import torch
from torch import nn
from torchaudio.transforms import MuLawDecoding
from src.models.decoders import LearnedUpsamplingDecoder1d
from src.models.encoders import LearnedDownsamplingEncoder1d
from src.models.quantizers import VanillaVectorQuantizer, GroupVectorQuantizer
from src.models.wavenet import WaveNet
from src.params import Params, global_params


class VQVAEVC(nn.Module):

    def __init__(self, params: Params):
        super(VQVAEVC, self).__init__()
        self.mu_law_decoder = MuLawDecoding(global_params.MU_QUANTIZATION_CHANNELS)
        self.encoder = LearnedDownsamplingEncoder1d(1, params.ENCODER_HIDDEN_CHANNELS, params.EMBEDDING_DIM,
                                                    params.ENCODER_KERNEL_SIZE, params.ENCODER_DOWNSAMPLING_STEPS)
        if params.VECTOR_QUANTIZER == "VanillaVectorQuantizer":
            self.quantizer = VanillaVectorQuantizer(params.EMBEDDING_DIM, params.NUM_EMBEDDINGS)
        elif params.VECTOR_QUANTIZER == "GroupVectorQuantizer":
            self.quantizer = GroupVectorQuantizer(params.EMBEDDING_DIM, params.NUM_EMBEDDING_GROUPS,
                                                  params.NUM_EMBEDDINGS_PER_GROUP)
        self.decoder = LearnedUpsamplingDecoder1d(params.EMBEDDING_DIM, params.DECODER_HIDDEN_CHANNELS,
                                                  params.DECODER_OUT_CHANNELS, params.DECODER_KERNEL_SIZE,
                                                  params.DECODER_UPSAMPLING_STEPS)
        self.speaker_embeddings = nn.Embedding(params.NUM_SPEAKERS, params.SPEAKER_DIM)
        self.wavenet = WaveNet(1, params.WAVENET_RESIDUAL_CHANNELS, params.WAVENET_DILATION_CHANNELS,
                               params.WAVENET_SKIP_CHANNELS, global_params.MU_QUANTIZATION_CHANNELS,
                               params.WAVENET_DILATION_STEPS, params.WAVENET_REPEATS,
                               use_local_conditioning=True, in_channels_local_condition=params.DECODER_OUT_CHANNELS,
                               use_global_conditioning=True, in_features_global_condition=params.SPEAKER_DIM)
        self.receptive_field_size = self.wavenet.receptive_field_size

    def forward(self, audios: torch.Tensor, speakers: torch.Tensor) -> dict[str, torch.Tensor]:
        n_samples = audios.size(-1)
        audios = self.mu_law_decoder(audios)
        encodings = self.encoder(audios)
        embeddings = self.quantizer(encodings)
        # Straight through reparameterization trick
        st_embeddings = encodings + torch.detach(embeddings - encodings)
        decodings = self.decoder(st_embeddings)[:, :, :n_samples]
        speaker_embeddings = self.speaker_embeddings(speakers)
        reconstructions = self.wavenet(audios[:, :, :n_samples-1], decodings, speaker_embeddings)
        return {
            "reconstructions": reconstructions,
            "encodings": encodings,
            "embeddings": embeddings
        }
