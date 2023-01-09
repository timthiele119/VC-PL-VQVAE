from torch import nn


class VQVAELoss(nn.Module):

    def __init__(self, beta=0.25, reconstruction_loss_fn: str = "cross_entropy"):
        super(VQVAELoss, self).__init__()
        self.beta = beta
        if reconstruction_loss_fn == "cross_entropy":
            self.reconstruction_loss_fn = nn.CrossEntropyLoss()
        elif reconstruction_loss_fn == "mse":
            self.reconstruction_loss_fn = nn.MSELoss()
        self.embedding_loss_fn = nn.MSELoss()
        self.commitment_loss_fn = nn.MSELoss()

    def forward(self, inputs, reconstructions, encodings, embeddings):
        reconstruction_loss = self.reconstruction_loss_fn(reconstructions, inputs)
        embedding_loss = self.embedding_loss_fn(embeddings, encodings.detach())
        commitment_loss = self.commitment_loss_fn(encodings, embeddings.detach())
        return reconstruction_loss + embedding_loss + self.beta * commitment_loss
