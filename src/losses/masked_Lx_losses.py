import torch
import torch.nn as nn

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, input, target, mask):
        mask = torch.stack([mask] * input.shape[-1], dim=-1)
        input *= mask
        target *= mask
        loss = self.l1_loss(input, target)
        return loss.mean()
    
    
class MaskedL2Loss(nn.Module):
    def __init__(self):
        super(MaskedL2Loss, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward(self, input, target, mask):
        mask = torch.stack([mask] * input.shape[-1], dim=-1)
        input *= mask
        target *= mask
        loss = self.l2_loss(input, target)
        return loss.mean()