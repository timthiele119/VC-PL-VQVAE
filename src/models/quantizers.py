from abc import abstractmethod
import torch
from torch import nn


class VectorQuantizer(nn.Module):

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super(VectorQuantizer, self).__init__()
        self.D = embedding_dim
        self.K = num_embeddings
        self.codebook = torch.nn.Parameter((torch.rand(self.D, self.K) - 0.5) * 2 / self.K, requires_grad=True)

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        permutation = list(range(len(encodings.shape)))
        permutation.append(permutation.pop(permutation[1]))
        # [B x D x H x W] -> [B x H x W x D]  (or [B x D x T] -> [B x T x D]))
        encodings = torch.permute(encodings, tuple(permutation)).contiguous()
        shape_encodings = encodings.shape
        # [B x H x W x D] -> [BHW x D]  (or [B x T x D] -> [BT x D])
        encodings = encodings.view(-1, self.D)

        embeddings = self.quantize(encodings)

        # Recover original shape
        # [BHW x D] -> [B x H x W x D]  (or [BT x D] -> [B x T x D])
        embeddings = embeddings.view(shape_encodings)
        # [B x H x W x D] -> [B x D x H x W]  (or [B x T x D] -> [B x D x T])
        reverse_permutation = list(range(len(permutation)))
        reverse_permutation.insert(1, reverse_permutation.pop())
        embeddings = embeddings.permute(tuple(reverse_permutation)).contiguous()
        return embeddings

    @abstractmethod
    def quantize(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

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
        sqrd_norm_embeddings = torch.sum(self.codebook ** 2, dim=0, keepdim=True)
        dot_inputs_embeddings = inputs @ self.codebook
        distances = sqrd_norm_inputs - 2 * dot_inputs_embeddings + sqrd_norm_embeddings
        return distances


class VanillaVectorQuantizer(VectorQuantizer):

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        embedding_distances = self._get_embedding_distances(encodings)
        embedding_ids = torch.argmin(embedding_distances, dim=1)
        embedding_ids_one_hot = nn.functional.one_hot(embedding_ids, num_classes=self.K).type(torch.float32)
        embeddings = embedding_ids_one_hot @ self.codebook.T
        return embeddings


class GroupVectorQuantizer(VectorQuantizer):

    def __init__(self, embedding_dim: int, num_groups: int, num_embeddings_per_group: int):
        super(GroupVectorQuantizer, self).__init__(embedding_dim, num_groups * num_embeddings_per_group)
        self.D = embedding_dim
        self.K = num_groups
        self.M = num_embeddings_per_group
        self.N = self.M * self.K

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        embedding_distances = self._get_embedding_distances(encodings)
        group_embedding_distances = embedding_distances.view(encodings.size(0), self.K, self.M)
        group_embedding_avg_distances = torch.mean(group_embedding_distances, dim=-1)
        nearest_groups = torch.argmin(group_embedding_avg_distances, dim=-1)
        nearest_group_embeddings = self._get_nearest_group_embeddings(nearest_groups)
        nearest_group_embedding_weights = self._get_nearest_group_embedding_weights(nearest_groups,
                                                                                    group_embedding_distances)
        embeddings = torch.sum(nearest_group_embedding_weights * nearest_group_embeddings, dim=-1)
        return embeddings

    def _get_nearest_group_embeddings(self, nearest_groups: torch.Tensor) -> torch.Tensor:
        nearest_group_embeddings = torch.index_select(self.codebook.view(self.D, self.K, self.M), 1, nearest_groups)
        nearest_group_embeddings = nearest_group_embeddings.permute((1, 0, 2))
        return nearest_group_embeddings

    def _get_nearest_group_embedding_weights(self, nearest_groups: torch.Tensor,
                                             group_embedding_distances: torch.Tensor) -> torch.Tensor:
        indices = nearest_groups.view(-1, 1).unsqueeze(2).repeat(1, 1, self.M)
        nearest_group_embedding_distances = torch.gather(group_embedding_distances, 1, indices)
        weights = 1 / nearest_group_embedding_distances
        weights = weights / torch.sum(weights, dim=-1, keepdim=True)
        return weights
