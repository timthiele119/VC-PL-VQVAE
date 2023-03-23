import os.path

import torch.cuda
import yaml
import numpy as np

from src.data.datamodules import VCMelDataModule
from src.models.hle_vqvae_vc import HleVqVaeVc
from src.modules.speakers import SpeakerEmbedding
from src.modules.quantizers import EMAVectorQuantizer, AttentionVectorQuantizer
from src.modules.encoders import HleEncoder
from src.modules.decoders import HleDecoder
from src.losses.vqvae_losses import HierarchicalVqVaeLoss
from src.params import PROJECT_ROOT

PATH_DATA_CONFIC = os.path.join(PROJECT_ROOT, "config", "data", "vctk20", "vctk20mel-24kHZ.yml")
PATH_MODEL_CONFIC = os.path.join(PROJECT_ROOT, "config", "model", "hle-vqvae-vc.yml")

device = "cuda:2" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    #arrays = [np.random.randn(80, 492).T, np.random.randn(80, 239).T]
    #np.savez_compressed("test.npz", arrays=arrays)
    #ds = np.load("test.npz", allow_pickle=True)
    #print(ds["arrays"])

    with open(PATH_DATA_CONFIC) as f:
        data_config = yaml.safe_load(f)["data"]
    with open(PATH_MODEL_CONFIC) as f:
        model_config = yaml.safe_load(f)["model"]

    data_module = VCMelDataModule(**data_config)
    data_module.prepare_data()
    data_loader = data_module.train_dataloader()
    mel, wav, speaker, emotion = next(iter(data_loader))
    mel, wav, speaker, emotion = mel.to(device), wav.to(device), speaker.to(device), emotion.to(device)
    print(mel.shape)
    print(wav)
    print(speaker)
    print(emotion)

    speaker_embedding = SpeakerEmbedding(**model_config["speaker_embedding"]["init_args"])
    encoder_bot = HleEncoder(**model_config["encoder_bot"]["init_args"]).to(device)
    encoder_mid = HleEncoder(**model_config["encoder_mid"]["init_args"])
    encoder_top = HleEncoder(**model_config["encoder_top"]["init_args"])
    quantizer_bot = EMAVectorQuantizer(**model_config["quantizer_bot"]["init_args"])
    quantizer_mid = EMAVectorQuantizer(**model_config["quantizer_mid"]["init_args"])
    quantizer_top = EMAVectorQuantizer(**model_config["quantizer_top"]["init_args"])
    decoder_bot = HleDecoder(**model_config["decoder_bot"]["init_args"])
    decoder_mid = HleDecoder(**model_config["decoder_mid"]["init_args"])
    decoder_top = HleDecoder(**model_config["decoder_top"]["init_args"])
    learning_rate = model_config["learning_rate"]
    quantizer = AttentionVectorQuantizer(16, 512, 8).to(device)

    _, encodings = encoder_bot(mel)
    embeddings = quantizer(encodings)

    #model = HleVqVaeVc(speaker_embedding, encoder_bot, encoder_mid, encoder_top, quantizer_bot, quantizer_mid,
    #                   quantizer_top, decoder_bot, decoder_mid, decoder_top).to(device)
    #model.training = True

    #reconstructed_audio, encodings, embeddings = model(audio, speaker)

    #print(f"Original audio: {audio.shape}")
    #print(f"Reconstructed audio: {reconstructed_audio.shape}")
    #loss_fn = HierarchicalVqVaeLoss()
    #print(f"Loss Function: {loss_fn(audio, reconstructed_audio, encodings, embeddings)}")


