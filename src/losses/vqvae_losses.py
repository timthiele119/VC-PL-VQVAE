from torch import nn
from torch.nn import functional as F


class VQVAELoss(nn.Module):

    def __init__(self, beta=0.25):
        super(VQVAELoss, self).__init__()
        self.beta = beta

    def forward(self, inputs, reconstructions, encodings, embeddings):
        reconstruction_loss = F.mse_loss(reconstructions, inputs)
        embedding_loss = F.mse_loss(embeddings, encodings.detach())
        commitment_loss = F.mse_loss(encodings, embeddings.detach())
        return reconstruction_loss + embedding_loss + self.beta * commitment_loss
