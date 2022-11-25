from abc import abstractmethod
from src.models.base import BaseModule
import torch
from torch import nn


class VectorQuantizer(BaseModule):

    def __init__(self):
        super(VectorQuantizer, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.quantize(inputs)

    @abstractmethod
    def quantize(self, inputs: torch.Tensor) -> torch.Tensor:
        pass


class VanillaVectorQuantizer(VectorQuantizer):

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super(VanillaVectorQuantizer, self).__init__()
        self.D = embedding_dim
        self.K = num_embeddings
        self.embeddings = torch.nn.Parameter((torch.rand(self.D, self.K) - 0.5) * 2 / self.K, requires_grad=True)

    @classmethod
    def from_config(cls, config: dict):
        pass

    def quantize(self, inputs: torch.Tensor) -> tuple:
        permutation = list(range(len(inputs.shape)))
        permutation.append(permutation.pop(permutation[1]))
        # [B x D x H x W] -> [B x H x W x D]
        inputs = torch.permute(inputs, tuple(permutation)).contiguous()
        shape_inputs = inputs.shape
        # [B x H x W x D] -> [BHW x D]
        inputs = inputs.view(-1, self.D)

        # Quantize inputs
        embedding_distances = self._get_embedding_distances(inputs)
        embedding_indices = self._get_embedding_indices(embedding_distances)
        embedding_vectors = self._get_embedding_vectors(embedding_indices)

        # Recover original shape
        # [BHW x D] -> [B x H x W x D]
        embedding_vectors = embedding_vectors.view(shape_inputs)
        # [B x H x W x D] -> [B x D x H x W]
        reverse_permutation = list(range(len(permutation)))
        reverse_permutation.insert(1, reverse_permutation.pop())
        embedding_vectors = embedding_vectors.permute(tuple(reverse_permutation)).contiguous()

        return embedding_vectors, embedding_indices, embedding_distances

    def _get_embedding_distances(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes squared L2-norm between each input vector to each embedding vector
        Note: ||x - e||**2 = (x - e).T @ (x - e) = x.T @ x - 2 * x.T @ e + e.T @ e
        :param
            inputs: tensor of shape [BHW x D]
        :return
            distances: matrix of squared L2-distances [BHW x K]:
        """
        sqrd_norm_inputs = torch.sum(inputs ** 2, dim=1, keepdim=True)
        sqrd_norm_embeddings = torch.sum(self.embeddings ** 2, dim=0, keepdim=True)
        dot_inputs_embeddings = inputs @ self.embeddings
        distances = sqrd_norm_inputs - 2 * dot_inputs_embeddings + sqrd_norm_embeddings
        return distances

    def _get_embedding_indices(self, distances: torch.Tensor) -> torch.Tensor:
        """
        For each input vector the index of the closest embedding vector is returned
        :param
            distances: matrix of squared L2-distances [BHW x K]
        :return
            embedding_indices: vector of embedding indices [BHW x 1]
        """
        embedding_indices = torch.argmin(distances, dim=1)
        return embedding_indices

    def _get_embedding_vectors(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        """
        For each input vector, the embedding index is replaced with the corresponding embedding vector
        :param
            embedding_indices: vector of embedding indices [BHW x 1]
        :return
            embedding_vectors: matrix of embedding vectors [BHW x D]:
        """
        embedding_indices_one_hot = nn.functional.one_hot(embedding_indices, num_classes=self.K).type(torch.float32)
        embeddings = embedding_indices_one_hot @ self.embeddings.T
        return embeddings