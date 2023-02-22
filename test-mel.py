import torch
from torch.utils.data import DataLoader
from parallel_wavegan.utils import load_model
import pytorch_lightning as pl

from src.data.datasets import VCTKMelDataset
from src.data.utils import MelSpecCollateFn
from src.data.datamodules import VCMelDataModule
from src.models.hle_vqvae_vc import HleVqVaeVc
from src.models.speakers import SpeakerEmbedding
from src.models.quantizers import VanillaVectorQuantizer
from src.models.encoders import HleEncoder
from src.models.decoders import HleDecoder
from src.losses.vqvae_losses import HierarchicalVqVaeLoss


if __name__ == "__main__":
    data_module = VCMelDataModule(
        train_dataset_class_name="VCTKMelDataset",
        train_dataset_init_args={
            "root_dir": "./data/vctk20/vctk20mel-24kHZ-train",
            "dataset_specific_config": {
                "vctk_speaker_list": "./config/data/vctk20/split/speakers_20_list.txt",
                "vctk_audio_dir": "/scratch/sghosh/datasets/vctkall-voxceleb-24k",
                "vctk_relative_audio_path_list": "./config/data/vctk20/split/train_20_list.txt"
            },
            "sr": 24000,
            "n_fft": 2048,
            "hop_length": 300,
            "win_length": 1200,
            "n_mels": 80
        },
        val_dataset_class_name="VCTKMelDataset",
        val_dataset_init_args={
            "root_dir": "./data/vctk20/vctk20mel-24kHZ-val",
            "dataset_specific_config": {
                "vctk_speaker_list": "./config/data/vctk20/split/speakers_20_list.txt",
                "vctk_audio_dir": "/scratch/sghosh/datasets/vctkall-voxceleb-24k",
                "vctk_relative_audio_path_list": "./config/data/vctk20/split/val_20_list.txt"
            },
            "sr": 24000,
            "n_fft": 2048,
            "hop_length": 300,
            "win_length": 1200,
            "n_mels": 80
        },
        batch_size=64
    )
    data_module.prepare_data()
    data_loader = data_module.train_dataloader()
    audio, speaker = next(iter(data_loader))

    speaker_embedding = SpeakerEmbedding(20, 16)
    encoder_bot = HleEncoder(input_dim=80, output_latent_dim=32, output_dim=128, residual_dim=128, skip_dim=128, gate_dim=128, kernel_size=5, dilation_steps=3,
                             dilation_repeats=2)
    encoder_mid = HleEncoder(input_dim=128, output_latent_dim=32, output_dim=128, residual_dim=128, skip_dim=128, gate_dim=128, kernel_size=5, dilation_steps=3,
                             dilation_repeats=2)
    encoder_top = HleEncoder(input_dim=128, output_latent_dim=32, output_dim=0, residual_dim=128, skip_dim=128, gate_dim=128, kernel_size=5, dilation_steps=3,
                             dilation_repeats=2)
    quantizer_bot = VanillaVectorQuantizer(32, 128)
    quantizer_mid = VanillaVectorQuantizer(32, 128)
    quantizer_top = VanillaVectorQuantizer(32, 128)
    decoder_top = HleDecoder(input_dim=32, residual_dim=128, skip_dim=128, cond_dim=16, gate_dim=128, kernel_size=5,
                             dilation_steps=3, output_dim=128)
    decoder_mid = HleDecoder(input_dim=160, residual_dim=128, skip_dim=128, cond_dim=16, gate_dim=128, kernel_size=5,
                             dilation_steps=3, output_dim=128)
    decoder_bot = HleDecoder(input_dim=160, residual_dim=128, skip_dim=128, cond_dim=16, gate_dim=128, kernel_size=5,
                             dilation_steps=4, output_dim=80)

    model = HleVqVaeVc(speaker_emb=speaker_embedding, encoder_bot=encoder_bot, encoder_mid=encoder_mid,
                       encoder_top=encoder_top, quantizer_bot=quantizer_bot, quantizer_mid=quantizer_mid,
                       quantizer_top=quantizer_top, decoder_top=decoder_top, decoder_mid=decoder_mid,
                       decoder_bot=decoder_bot)
    reconstructed_audio, encodings, embeddings = model(audio, speaker)
    print(f"Original audio: {audio.shape}")
    print(f"Reconstructed audio: {reconstructed_audio.shape}")
    loss_fn = HierarchicalVqVaeLoss()
    print(f"Loss Function: {loss_fn(audio, reconstructed_audio, encodings, embeddings)}")


