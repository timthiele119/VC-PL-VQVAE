from typing import Dict, List, Union

import os
import torch
from torch import nn
from torch.nn import functional as F
          
from src.losses.perceptual_losses import *
from src.losses.vgg_perceptual_loss import *
from src.params import global_params


class HierarchicalVqVaeLoss(nn.Module):

    def __init__(self, beta: float = 0.25, reconstruction_loss_fn: str = "mse",
                 use_codebook_loss: bool = False):
        super(HierarchicalVqVaeLoss, self).__init__()
        self.beta = beta
        if reconstruction_loss_fn == "mse":
            self.reconstruction_loss_fn = nn.MSELoss()
        elif reconstruction_loss_fn == "cross_entropy":
            self.reconstruction_loss_fn = nn.CrossEntropyLoss()
        else:
            raise Exception(f"Reconstruction loss function \"{reconstruction_loss_fn}\" not known")
        self.codebook_loss_fn = nn.MSELoss()
        self.commitment_loss_fn = nn.MSELoss()
        self.use_codebook_loss = use_codebook_loss

    def forward(self, original: torch.Tensor, reconstruction: torch.Tensor, encodings: List[torch.Tensor],
                embeddings: List[torch.Tensor]):
        reconstruction_loss = self.reconstruction_loss_fn(reconstruction, original)
        codebook_loss, commitment_loss = 0, 0
        for embedding, encoding in zip(embeddings, encodings):
            codebook_loss += self.codebook_loss_fn(embedding, encoding.detach()) if self.use_codebook_loss else 0
            commitment_loss += self.commitment_loss_fn(encoding, embedding.detach())
        return reconstruction_loss, codebook_loss, self.beta * commitment_loss 
    


class PerceptualVqVaeVcLoss(nn.Module):

    def __init__(
        self, 
        loss_weights: Dict,  
        loss_computation_epochs: Dict = None,
        vocoder = None,
    ):
        super(PerceptualVqVaeVcLoss, self).__init__()
        
        # Weights
        loss_weights = loss_weights if loss_weights else {}
        loss_weights["reconstruction_loss"] = loss_weights["reconstruction_loss"] if "reconstruction_loss" in loss_weights else 1.0
        loss_weights["codebook_loss"] = loss_weights["codebook_loss"] if "codebook_loss" in loss_weights else 0.0  # For EMA updates
        loss_weights["commitment_loss"] = loss_weights["commitment_loss"] if "commitment_loss" in loss_weights else 0.25
        loss_weights["f0_consistency_loss"] = loss_weights["f0_consistency_loss"] if "f0_consistency_loss" in loss_weights else 0.0
        loss_weights["mel_diversification_loss"] = loss_weights["mel_diversification_loss"] \
            if "mel_diversification_loss" in loss_weights else 0.0
        loss_weights["f0_diversification_loss"] = loss_weights["f0_diversification_loss"] \
            if "f0_diversification_loss" in loss_weights else 0.0
        loss_weights["formant_mse_loss"] = loss_weights["formant_mse_loss"] if "formant_mse_loss" in loss_weights and not loss_weights["formant_mse_loss"] is None else 0.0
        loss_weights["mos_loss"] = loss_weights["mos_loss"] if "mos_loss" in loss_weights and not loss_weights["mos_loss"] is None else 0.0
        loss_weights["codec_loss"] = loss_weights["codec_loss"] if "codec_loss" in loss_weights and not loss_weights["codec_loss"] is None else 0.0
        loss_weights["WavLM_loss"] = loss_weights["WavLM_loss"] if "WavLM_loss" in loss_weights and not loss_weights["WavLM_loss"] is None else 0.0
        loss_weights["vgg_loss"] = loss_weights["vgg_loss"] if "vgg_loss" in loss_weights and not loss_weights["vgg_loss"] is None else 0.0
        loss_weights["F0_loss"] = loss_weights["F0_loss"] if "F0_loss" in loss_weights and not loss_weights["F0_loss"] is None else 0.0

        assert loss_weights["reconstruction_loss"] > 0.0
        print(f"LOSS WEIGHTS: {loss_weights}")
        self.loss_weights = loss_weights


        # Loss computation epochs
        lce = loss_computation_epochs if loss_computation_epochs else {}
        lce["reconstruction_loss"] = lce["reconstruction_loss"] if "reconstruction_loss" in lce else 0
        lce["codebook_loss"] = lce["codebook_loss"] if "codebook_loss" in lce else 0
        lce["commitment_loss"] = lce["commitment_loss"] if "commitment_loss" in lce else 0
        lce["f0_consistency_loss"] = lce["f0_consistency_loss"] if "f0_consistency_loss" in lce else 0
        lce["mel_diversification_loss"] = lce["mel_diversification_loss"] \
            if "mel_diversification_loss" in lce else 0
        lce["f0_diversification_loss"] = lce["f0_diversification_loss"] \
            if "f0_diversification_loss" in lce else 0
        lce["formant_mse_loss"] = lce["formant_mse_loss"] if "formant_mse_loss" in lce else 0
        lce["mos_loss"] = lce["mos_loss"] if "mos_loss" in lce and not lce["mos_loss"] is None else 0
        lce["codec_loss"] = lce["codec_loss"] if "codec_loss" in lce and not lce["codec_loss"] is None else 0
        lce["WavLM_loss"] = lce["WavLM_loss"] if "WavLM_loss" in lce and not lce["WavLM_loss"] is None else 0
        lce["vgg_loss"] = lce["vgg_loss"] if "vgg_loss" in lce and not lce["vgg_loss"] is None else 0
        lce["F0_loss"] = lce["F0_loss"] if "F0_loss" in lce and not lce["F0_loss"] is None else 0
        print(f"LOSS COMPUTATION EPOCHS: {lce}")
        self.lce = lce
       
        # init perceptual losses
        if loss_weights["mos_loss"]>0:
            self.mos_loss = wvmos_loss(vocoder)
        if loss_weights["codec_loss"]>0:
            checkpoint_path = "lightning_logs/version_73_codec_ssim/checkpoints"
            self.codec_loss = codec_activation_loss(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))

        if loss_weights["vgg_loss"]>0:
            self.vgg_loss = VGGPerceptualLoss()
        if loss_weights['WavLM_loss']>0:
            self.WavLM_loss = WAVLM_loss(vocoder)
        if loss_weights['formant_mse_loss']>0:
            checkpoint_path = global_params.FORMANT_MODEL_CHECKPOINT_PATH
            formant_model_path = os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])
            self.formant_mse_loss = Transformer_FormantMSELoss(formant_model_path)

        if loss_weights['F0_loss']>0:
            self.F0_loss = F0_Loss(vocoder)

    def forward(
            self,
            original: torch.Tensor,
            reconstruction: torch.Tensor,
            conversion: torch.Tensor,
            encodings: Union[torch.Tensor, List[torch.Tensor]],
            embeddings: Union[torch.Tensor, List[torch.Tensor]],
            f0_original: torch.Tensor,
            f0_reconstruction: torch.Tensor,
            f0_conversion: torch.Tensor,
            current_epoch: int
    ) -> torch.Tensor:
        if isinstance(encodings, torch.Tensor):
            encodings = [torch.Tensor]
        if isinstance(embeddings, torch.Tensor):
            embeddings = [torch.Tensor]
        assert len(encodings) == len(embeddings)
        self.reconstruction_loss = F.mse_loss(reconstruction, original)
        self.codebook_loss, self.commitment_loss = 0, 0
        for encoding, embedding in zip(encodings, embeddings):
            self.codebook_loss += F.mse_loss(embedding, encoding.detach())
            self.commitment_loss += F.mse_loss(encoding, embedding.detach())
        self.mel_diversification_loss = - F.l1_loss(conversion, original) \
            if self.loss_weights["mel_diversification_loss"] > 0.0 and conversion is not None else 0.0
        self.f0_consistency_loss = 0.0
        if self.loss_weights["f0_consistency_loss"] > 0.0:
            f0_original_normed = f0_original / torch.mean(f0_original, 1, keepdim=True)
            f0_reconstruction_normed = f0_reconstruction / torch.mean(f0_reconstruction, 1, keepdim=True)
            f0_consistency_loss = F.l1_loss(f0_reconstruction_normed, f0_original_normed)
        self.f0_diversification_loss = 0.0
        if self.loss_weights["f0_diversification_loss"] > 0.0:
            voiced = f0_original >= 80.0
            self.f0_diversification_loss = - F.l1_loss(f0_conversion[voiced], f0_original[voiced])
        
        # compute vanilla losses
        loss_dict = {
            "reconstruction_loss": self.loss_weights["reconstruction_loss"] * self.reconstruction_loss,
            "codebook_loss": self.loss_weights["codebook_loss"] * self.codebook_loss,
            "commitment_loss": self.loss_weights["commitment_loss"] * self.commitment_loss,
            "mel_diversification_loss" : self.loss_weights["mel_diversification_loss"] * self.mel_diversification_loss,
            "f0_consistency_loss": self.loss_weights["f0_consistency_loss"] * self.f0_consistency_loss,
            "f0_diversification_loss": self.loss_weights["f0_diversification_loss"] * self.f0_diversification_loss
        }

        # compute and weight all used perceptual losses
        if current_epoch >= self.lce["formant_mse_loss"] and self.loss_weights['formant_mse_loss']>0:
            loss_dict["formant_mse_loss"] = self.loss_weights["formant_mse_loss"] * self.formant_mse_loss(original, reconstruction)
        if current_epoch >= self.lce["mos_loss"] and self.loss_weights['mos_loss']>0:
            loss_dict["mos_loss"] = self.loss_weights["mos_loss"] * self.mos_loss(original, reconstruction)
        if current_epoch >= self.lce["codec_loss"] and self.loss_weights['codec_loss']>0:
            loss_dict["codec_loss"] = self.loss_weights["codec_loss"] * self.codec_loss(original, reconstruction)
        if current_epoch >= self.lce["WavLM_loss"] and self.loss_weights['WavLM_loss']>0:
            loss_dict["WavLM_loss"] = self.loss_weights["WavLM_loss"] * self.WavLM_loss(original, reconstruction)
        if current_epoch >= self.lce["vgg_loss"] and self.loss_weights['vgg_loss']>0:
            loss_dict["vgg_loss"] = self.loss_weights["vgg_loss"] * self.vgg_loss(original.unsqueeze(dim=1), reconstruction.unsqueeze(dim=1))
        if current_epoch >= self.lce["F0_loss"] and self.loss_weights['F0_loss']>0:
            loss_dict["F0_loss"] = self.loss_weights["F0_loss"] * self.F0_loss(original, reconstruction)

        for key in loss_dict.keys():
            if key in self.lce.keys():
                loss_dict[key] = loss_dict[key] if current_epoch >= self.lce[key] else 0.0                
            else:
                raise NotImplementedError(f"Loss function {key} is not supported.")
        loss_dict["total_loss"] = sum(loss_dict.values())
        
        print(f"Loss dict: {loss_dict} \n")
        
        return loss_dict