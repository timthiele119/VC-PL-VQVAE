from wavlm.WavLM import WavLM, WavLMConfig
import torch
import logging
import torchaudio
from pathlib import Path
from torch import Tensor
import torchaudio.transforms as T
import torch.nn as nn


class FeatureExtractor(nn.Module):

    def __init__(self,
                 wavlm: WavLM,
                 device: str,
                 sr=16000):
        super(FeatureExtractor, self).__init__()
        self.wavlm = wavlm.eval()
        self.device = device
        self.sr = sr

    @torch.inference_mode()
    def get_features(self, path, vad_trigger_level=0, layer=6):
        """Returns features of `path` waveform as a tensor of shape (seq_len, dim), optionally perform VAD trimming
        on start/end with `vad_trigger_level`.
        """
        # load audio
        if type(path) in [str, Path]:
            x, sr = torchaudio.load(path, normalize=True)
        else:
            x: Tensor = path
            sr = self.sr
            if x.dim() == 1: x = x[None]

        if not sr == self.sr:
            print(f"resample {sr} to {self.sr} in {path}")
            x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=self.sr)
            sr = self.sr

        # trim silence from front and back
        if vad_trigger_level > 1e-3:
            transform = T.Vad(sample_rate=sr, trigger_level=vad_trigger_level)
            x_front_trim = transform(x)
            # original way, disabled because it lacks windows support
            # waveform_reversed, sr = apply_effects_tensor(x_front_trim, sr, [["reverse"]])
            waveform_reversed = torch.flip(x_front_trim, (-1,))
            waveform_reversed_front_trim = transform(waveform_reversed)
            waveform_end_trim = torch.flip(waveform_reversed_front_trim, (-1,))
            # waveform_end_trim, sr = apply_effects_tensor(
            #    waveform_reversed_front_trim, sr, [["reverse"]]
            # )
            x = waveform_end_trim

        # extract the representation of each layer
        wav_input_16khz = x.to(self.device)
        # use fastpath
        features = \
                self.wavlm.extract_features(wav_input_16khz, output_layer=layer, ret_layer_results=False)[0]

        return features.squeeze(0)


def wavlm_large(pretrained=True, progress=True, device='cuda') -> WavLM:
    """Load the WavLM large checkpoint from the original paper. See https://github.com/microsoft/unilm/tree/master/wavlm for details. """
    if torch.cuda.is_available() == False:
        if str(device) != 'cpu':
            logging.warning(f"Overriding device {device} to cpu since no GPU is available.")
            device = 'cpu'
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt",
        map_location=device,
        progress=progress
    )

    cfg = WavLMConfig(checkpoint['cfg'])
    device = torch.device(device)
    model = WavLM(cfg)
    if pretrained:
        model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"WavLM-Large loaded with {sum([p.numel() for p in model.parameters()]):,d} parameters.")
    return model



if __name__ == '__main__':

    pretrained = True
    progress = True
    device = 'cuda'

    wavlm = wavlm_large(pretrained, progress, device)
    print(wavlm)
    wav_input_16khz = '/cache/sghosh/IS2024/DDSP/supplementary_v2/stuttering/3_F2F/Original.wav'
    fe = FeatureExtractor(wavlm=wavlm, device=device)
    speaker_features = fe.get_features(wav_input_16khz)