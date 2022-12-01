from abc import abstractmethod
import torch
from torch.nn import functional as F


class CodebookCollapseHandler:

    def __init__(self, codebook: torch.Tensor):
        self.codebook = codebook
        self.embedding_dim = codebook.shape[0]
        self.num_embeddings = codebook.shape[1]

    @abstractmethod
    def step(self, data: dict):
        pass


class RandomRestarts(CodebookCollapseHandler):
    """
    Implements random restarts as suggested in https://arxiv.org/abs/2005.00341 in order to prevent codebook collapse.
    It replaces a codebook vector with a random encoding from the current batch if the avg. per-sample usage frequency
    of the codebook vector falls below a given threshold.
    """
    def __init__(self, codebook: torch.Tensor, usage_threshold: float = 0.01, alpha: float = 0.1, device: str = "cpu"):
        """
        :param codebook: Codebook of VQ-VAE
        :param usage_threshold: Per sample usage frequency threshold
        :param alpha: EMA hyperparameter used to update the avg. usage frequencies of the codebook vectors
        :param device: cpu / cuda device
        """
        super(RandomRestarts, self).__init__(codebook)
        self.embedding_usage_freq = torch.zeros(self.num_embeddings, device=device)
        self.usage_threshold = usage_threshold
        self.alpha = alpha
        self.t = 1
        self.device = device

    def step(self, batch_data: dict):
        encodings, embedding_ids = batch_data['encodings'], batch_data['embedding_ids']
        batch_size = encodings.shape[0]
        self._update_embedding_usage_freq(embedding_ids, batch_size)
        self._update_codebook(encodings)

    def _update_embedding_usage_freq(self, embedding_ids: torch.Tensor, batch_size: int):
        with torch.no_grad():
            embedding_ids = F.one_hot(embedding_ids, self.num_embeddings)
            current_embedding_usage_freq = torch.sum(embedding_ids, dim=0) / batch_size
            self.embedding_usage_freq = (self.alpha * self.embedding_usage_freq +
                                         (1 - self.alpha) * current_embedding_usage_freq) / (1 - self.alpha ** self.t)
            self.t += 1

    def _update_codebook(self, encodings: torch.Tensor):
        with torch.no_grad():
            below_threshold = torch.where(self.embedding_usage_freq <= self.usage_threshold)[0]
            n_below_threshold = len(below_threshold)
            if n_below_threshold > 0:
                print(f"below threshold: {n_below_threshold}")
                encodings = self._reshape_encodings(encodings)
                replace_encoding_indices = torch.multinomial(torch.ones(encodings.shape[0]), n_below_threshold, replacement=True)
                replace_encodings = encodings[replace_encoding_indices].T
                replace_encodings += torch.randn_like(replace_encodings) * 0.0001
                self.codebook[:,below_threshold] = replace_encodings

    def _reshape_encodings(self, encodings: torch.Tensor) -> torch.Tensor:
        permutation = list(range(len(encodings.shape)))
        permutation.append(permutation.pop(permutation[1]))
        # [B x D x H x W] -> [B x H x W x D]
        encodings = torch.permute(encodings, tuple(permutation)).contiguous()
        # [B x H x W x D] -> [BHW x D]
        return encodings.view(-1, self.embedding_dim)