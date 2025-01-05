import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from src.external.formants.transformer_modules import Encoder, Decoder, RegressionHead
from src.losses.masked_Lx_losses import MaskedL1Loss, MaskedL2Loss


class TimeSeriesTransformer(pl.LightningModule):
    
    def __init__(
        self,
        src_vocab_size=80,
        src_pad_idx=0,
        trg_vocab_size=3,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        max_length=10000,
        hidden_size_reg_head=256,
        loss="L2",
        learning_rate=0.001,
        cyclic_lr=None,
        train_dataloader=None,
        val_dataloader=None,
    ):
        super(TimeSeriesTransformer, self).__init__()
        
        torch.autograd.set_detect_anomaly(True)
        
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.cyclic_lr = cyclic_lr
        if loss == "L1":
            self.loss_fn = MaskedL1Loss()
        elif loss == "L2":
            self.loss_fn = MaskedL2Loss()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length
        )

        self.regression_head = RegressionHead(
            embed_size, 
            hidden_size_reg_head,
            no_outputs=trg_vocab_size
        )
        
        self.src_pad_idx = src_pad_idx       
        
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader


    def make_src_mask(self, src):
        
        if src.dim() == 3:
            if src.shape[-1] != 80:
                src = torch.permute(src, dims=(0, 2, 1))
            src_copy = src.sum(dim=-1)
        else:
            src_copy = src
        src_mask = (src_copy != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        
        return src_mask

    
    def forward(self, src):
        
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask, self.device)
        out = self.regression_head(enc_src)
        
        return out
    
    
    def training_step(self, batch, batch_idx):
        
        mels, f1, f2, f3, mask = batch
        target = torch.cat((f1, f2, f3), dim=-1)
        trf_output = self(mels)
        loss = self.loss_fn(trf_output, target, mask)
        self.log("train_loss", loss)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        mels, f1, f2, f3, mask = batch
        target = torch.cat((f1, f2, f3), dim=-1)
        trf_output = self(mels)
        loss = self.loss_fn(trf_output, target, mask)
        self.log("val_loss", loss)
        
        return loss
    
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        
        if self.cyclic_lr is None:
            return ({"optimizer": optimizer})
        
        else:
            lr_scheduler = {
                'scheduler': torch.optim.lr_scheduler.CyclicLR(
                    optimizer, 
                    base_lr=self.cyclic_lr[0], 
                    max_lr=self.cyclic_lr[1], 
                    cycle_momentum=False
                ),
                'name': 'cyclic_lr'
            }
            return [optimizer], [lr_scheduler]
    
    
    @torch.no_grad()
    def inference(self, mels):
        
        trf_output = self(mels)
        f1_pred, f2_pred, f3_pred = trf_output[:, :, 0], trf_output[:, :, 1], trf_output[:, :, 2]
        
        return f1_pred, f2_pred, f3_pred
    
    
    def train_dataloader(self):
        return self.train_dataloader
    
    def val_dataloader(self):
        return self.val_dataloader
    

class Transformer(pl.LightningModule):
    
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        max_length=10000,
        loss="L2",
        learning_rate=0.001,
    ):
        super(Transformer, self).__init__()
        
        torch.autograd.set_detect_anomaly(True)
        
        self.max_length = max_length
        self.learning_rate = learning_rate
        if loss == "L1":
            self.loss_fn = MaskedL1Loss()
        elif loss == "L2":
            self.loss_fn = MaskedL2Loss()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx


    def make_src_mask(self, src):
        
        if src.dim() == 3:
            src_copy = src.sum(dim=-1)
        else:
            src_copy = src
        src_mask = (src_copy != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        
        return src_mask


    def make_trg_mask(self, trg):
        
        if trg.dim() == 3:
            N, trg_len, feat_dim = trg.shape
        else:
            N, trg_len = trg.shape
        trg_mask = torch.tril(
            torch.ones((trg_len, trg_len))
        ).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask

    
    def forward(self, src, trg):
        
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask, self.device)
        out = self.decoder(trg, enc_src, src_mask, trg_mask, self.device)
        
        return out
    
    
    def training_step(self, batch, batch_idx):
        
        mels, f1, f2, f3, mask = batch
        mels = mels[:, :-2, :].clone() # cut out XOS token paddings
        target = torch.cat((f1, f2, f3), dim=-1)
        target_input = target[:, :-1, :].clone()
        trf_output = self(mels, target_input)
        target_ground_truth = target[:, 1:, :].clone()
        target_loss_mask = mask[:, 1:].clone()
        loss = self.loss_fn(trf_output, target_ground_truth, target_loss_mask)
        self.log("train_loss", loss)
        
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        
        mels, f1, f2, f3, mask = batch
        mels = mels[:, :-2, :].clone() # cut out XOS token paddings
        target = torch.cat((f1, f2, f3), dim=-1)
        target_input = target[:, :-1, :].clone()
        trf_output = self(mels, target_input)
        target_ground_truth = target[:, 1:, :].clone()
        target_loss_mask = mask[:, 1:].clone()
        loss = self.loss_fn(trf_output, target_ground_truth, target_loss_mask)
        self.log("val_loss", loss)
        
        return loss
    
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return ({
            "optimizer": optimizer
        })
    
    
    @torch.no_grad()
    def inference(self, mels):
        """
        Not fully developed and debugged yet.
        """
        
        batch_size = mels.shape[0]
        formant_dim = 3
        
        SOS_value = -1
        EOS_value = -2
        SOS_token = torch.Tensor(np.full(shape=(batch_size, 1, formant_dim), fill_value=SOS_value))
        EOS_token = torch.Tensor(np.full(shape=(batch_size, 1, formant_dim), fill_value=EOS_value))
        print(f"Using Token: SOS == {SOS_value} and EOS == {EOS_value}")
        
        input_ = SOS_token
        
        for _ in range(min(self.max_length, mels.shape[1])):
            pred = self(mels, input_)
            next_item = pred[:, -1, :].unsqueeze(dim=1)
            input_ = torch.concat([input_, next_item], dim=1)
            
            if torch.equal(next_item, EOS_token):
                break
        
        trf_output = input_
        f1_pred, f2_pred, f3_pred = trf_output[:, :, 0], trf_output[:, :, 1], trf_output[:, :, 2]
        
        return f1_pred, f2_pred, f3_pred