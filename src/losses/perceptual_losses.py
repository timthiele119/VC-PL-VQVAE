import os
import torch
from torch import nn
import pytorch_lightning as pl
import pyworld as pw
from src.external.formants.transformer import TimeSeriesTransformer
from src.external.wvmos.wvmos import get_wvmos
from src.external.wavlm.WavLM import WavLM, WavLMConfig
from src.params import global_params
from parallel_wavegan.utils import load_model


class MOS_activation_loss(nn.Module):
    def __init__(self, model_path: str) -> None:
        super(MOS_activation_loss, self).__init__()
        self.model_path = model_path
        self.model = MBNet.load_from_checkpoint(self.model_path)
        self.activation = {}
        self.mse = nn.MSELoss()

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook
    
    def forward(self, original, reconstruction):
        self.model.mean_net_conv.register_forward_hook(self.get_activation('conv'))
        self.model.mean_net_dnn.register_forward_hook(self.get_activation('dnn'))
        #self.model.mean_net_rnn.register_forward_hook(self.get_activation('rnn'))

        output_original = self.model.only_mean_inference(original)        
        origin_activation_conv = self.activation['conv']
        origin_activation_dnn = self.activation['dnn']
        #origin_activation_rnn = self.activation['rnn']

        output_reconstruction = self.model.only_mean_inference(reconstruction)
        recon_activation_conv = self.activation['conv']
        recon_activation_dnn = self.activation['dnn']
        #recon_activation_rnn = self.activation['rnn']

        weight = len(self.activation.keys())

        loss = 1 / (weight + 1) * (self.mse(output_original, output_reconstruction)
                         + self.mse(origin_activation_conv, recon_activation_conv) 
                         + self.mse(origin_activation_dnn, recon_activation_dnn) )
                         #+ self.mse(origin_activation_rnn['rnn'], recon_activation_rnn['rnn']))
        
        return loss


class wvmos_loss(nn.Module):
    def __init__(self, mos_net=None, vocoder=None, device: str="cuda") -> None:
        super(wvmos_loss, self).__init__()
        self.sr = 16000
        self.activation = {}
        self.mse = nn.MSELoss()
        self.device = torch.device(device)

        if vocoder is None:
            self.vocoder = load_model(global_params.PATH_HIFIGAN_PARAMS).to(device).requires_grad_(True)
        else:
            self.vocoder = vocoder
            print("vocoder is passed")

        if mos_net is None:
            self.model = get_wvmos(cuda=False)
            self.model = self.model.to(device)
        else:
            self.model = mos_net.to(device)
            print("mos_net is passed")


class WAVLM_loss(nn.Module):
    def __init__(self, vocoder, checkpoint_path, device = torch.device('cuda')) -> None:
        super(WAVLM_loss, self).__init__()
        self.used_device = device
        # load the pre-trained checkpoints
        self.checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.used_device))
        self.cfg = WavLMConfig(self.checkpoint['cfg'])
        self.WAVLM = WavLM(self.cfg).to(self.used_device)
        self.WAVLM.load_state_dict(self.checkpoint['model'])
        self.WAVLM = self.WAVLM.to(self.used_device)
        for param in self.WAVLM.parameters():
            param.requires_grad = False
        self.WAVLM.eval()
        self.mse = nn.MSELoss()
        self.vocoder = vocoder.to(self.used_device)


    def forward(self, original, reconstruction):

        original_wav = self.vocoder(original).squeeze()
        rec_wav = self.vocoder(reconstruction).squeeze()

        if self.cfg.normalize:
            original_wav = torch.nn.functional.layer_norm(original_wav , original_wav.shape).to(self.used_device)
            rec_wav = torch.nn.functional.layer_norm(rec_wav , rec_wav.shape).to(self.used_device)

        SPEAKER_INFORMATION_LAYER = 6
        original_features = self.WAVLM.extract_features(original_wav, output_layer=SPEAKER_INFORMATION_LAYER, ret_layer_results=False)[0]
        reconstruction_features = self.WAVLM.extract_features(rec_wav, output_layer=SPEAKER_INFORMATION_LAYER, ret_layer_results=False)[0]

        loss = self.mse(original_features, reconstruction_features) 
        return loss
    
    
class Transformer_FormantMSELoss(nn.Module):
    
    def __init__(
        self, 
        model_path: str, 
        model_specs: dict | None = None, 
        model: pl.LightningModule = TimeSeriesTransformer
    ):
        super(Transformer_FormantMSELoss, self).__init__()
        self.model_path = model_path
        self.model = model.load_from_checkpoint(self.model_path)
        print(f"Transformer Model in use: {self.model}")
        self.model_specs = model_specs
        self.activation = {}
        self.loss_fn = nn.MSELoss()
    
    def get_activation(self, name):
        def _hook(model, input, output):
            if isinstance(output, tuple):
                self.activation[name] = output[0].detach()
            else:
                self.activation[name] = output.detach()
        return _hook
    
    def forward(self, original, reconstruction) -> torch.Tensor:
        loss = 0
        avg_weight = 0
        
        # Use formant model output
        pred_orig = self.model(original)
        pred_recon = self.model(reconstruction)
        loss += self.loss_fn(pred_orig, pred_recon)
        
        weighted_loss =  1 / (avg_weight + 1) * loss
        return weighted_loss