from torch import nn
from torch.nn import functional as F


class VQVAELoss(nn.Module):

    def __init__(self, beta=0.25, reconstruction_loss_fn: str = "cross_entropy"):
        super(VQVAELoss, self).__init__()
        self.beta = beta
        if reconstruction_loss_fn == "cross_entropy":
            self.reconstruction_loss_fn = F.cross_entropy
        elif reconstruction_loss_fn == "mse":
            self.reconstruction_loss_fn = F.mse_loss

    def forward(self, inputs, reconstructions, encodings, embeddings):
        reconstruction_loss = self.reconstruction_loss_fn(reconstructions, inputs)
        embedding_loss = F.mse_loss(embeddings, encodings.detach())
        commitment_loss = F.mse_loss(encodings, embeddings.detach())
        return reconstruction_loss + embedding_loss + self.beta * commitment_loss
