from abc import abstractmethod

import torch
from torch import nn


class VectorQuantizer(nn.Module):
    """
    Vector quantizer base module

    Args:
        embedding_dim (int): Dimensionality of each codebook vector
        num_embeddings (int): Number of codewords in the codebook
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super(VectorQuantizer, self).__init__()
        self.D = embedding_dim
        self.K = num_embeddings
        self.codebook = torch.nn.Parameter(torch.Tensor(self.D, self.K).uniform_(-1/(self.K*1e10), 1/(self.K*1e10)),
                                           requires_grad=True)

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

    def _get_codeword_distances(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        Computes squared L2-norm between each input vector to each codebook vector
        Note: ||x - e||**2 = (x - e).T @ (x - e) = x.T @ x - 2 * x.T @ e + e.T @ e

        Args:
            encodings (torch.Tensor): Tensor of shape [BHW x D] or [BT x D]

        Returns:
            distances (torch.Tensor: matrix of squared L2-distances [BHW x K] or [BT x K]:
        """
        sqrd_norm_inputs = torch.sum(encodings ** 2, dim=1, keepdim=True)
        sqrd_norm_embeddings = torch.sum(self.codebook ** 2, dim=0, keepdim=True)
        dot_inputs_embeddings = encodings @ self.codebook
        distances = sqrd_norm_inputs - 2 * dot_inputs_embeddings + sqrd_norm_embeddings
        return distances


class VanillaVectorQuantizer(VectorQuantizer):
    """
        Vector quantizer from the paper https://arxiv.org/abs/1711.00937

        Args:
            embedding_dim (int): Dimensionality of each codeword
            num_embeddings (int): Number of codewords in the codebook
            ema_gamma (float): Decay factor for EMA updates in range (0, 1). If None, EMA updates will not be performed.
                               (default 0.99)
            random_restarts_freq (int): If set to i > 0, it will perform "random restarts"
                                        (https://arxiv.org/abs/2005.00341) and replace unused codewords in the codebook
                                        with one random encoding vector + noise every ith training step in order to
                                        prevent codebook collapse. If None, random restarts will will not be performed.
                                        (default 64)

        """

    def __init__(self, embedding_dim: int, num_embeddings: int, ema_gamma: float = 0.99, random_restarts_freq: int = 64):
        super(VanillaVectorQuantizer, self).__init__(embedding_dim, num_embeddings)
        self._ema_gamma = ema_gamma
        self._ema_active = ema_gamma is not None
        if not self._ema_active:
            self.codebook.requires_grad = False
        self._mean_usage_counts = nn.Parameter(torch.Tensor(num_embeddings), requires_grad=False)
        nn.init.constant_(self._mean_usage_counts, 1e-5)
        self._total_usage_counts = nn.Parameter(torch.Tensor(num_embeddings), requires_grad=False)
        nn.init.constant_(self._total_usage_counts, 0.0)
        self._mean_encodings = nn.Parameter(torch.clone(self.codebook.detach()), requires_grad=False)
        self._train_step = 1
        self._random_restarts_freq = random_restarts_freq
        self._random_restarts_active = random_restarts_freq is not None

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        self._random_restarts(encodings)
        codeword_distances = self._get_codeword_distances(encodings)
        codeword_ids = torch.argmin(codeword_distances, dim=1)
        codeword_ids_one_hot = nn.functional.one_hot(codeword_ids, num_classes=self.K).type(torch.float32)
        embeddings = codeword_ids_one_hot @ self.codebook.T
        self._update_stats(encodings, codeword_ids_one_hot)
        self._ema_update()
        return embeddings

    def _update_stats(self, encodings: torch.Tensor, codeword_ids_one_hot: torch.Tensor):
        if self.training:
            self._train_step += 1
            batch_usage_counts = torch.sum(codeword_ids_one_hot, dim=0)
            if self._random_restarts_active:
                self._total_usage_counts.data += batch_usage_counts
            if self._ema_active:
                self._mean_usage_counts.data = self._ema_gamma * self._mean_usage_counts + \
                                               (1 - self._ema_gamma) * batch_usage_counts
                self._mean_encodings.data = self._ema_gamma * self._mean_encodings + \
                                            (1 - self._ema_gamma) * encodings.T @ codeword_ids_one_hot

    def _random_restarts(self, encodings: torch.Tensor):
        if self.training and self._train_step % self._random_restarts_freq == 0 and self._random_restarts_active:
            with torch.no_grad():
                unused_codewords = self._total_usage_counts < 1.0
                num_unused_codewords = sum(unused_codewords)
                if num_unused_codewords > 0:
                    print(f"Number of unused codewords: {num_unused_codewords}")
                    replace_encoding_indices = torch.multinomial(torch.ones(encodings.size(0)), num_unused_codewords,
                                                                 replacement=True)
                    replace_encodings = encodings[replace_encoding_indices].T
                    replace_encodings += torch.rand_like(replace_encodings) * 1e-6
                    self.codebook[:, unused_codewords] = replace_encodings
                    if self._ema_active:
                        self._mean_encodings[:, unused_codewords] = replace_encodings
                        self._mean_usage_counts[unused_codewords] = 1e-5
                nn.init.constant_(self._total_usage_counts, 0.0)

    def _ema_update(self):
        if self._ema_active:
            self.codebook.data = self._mean_encodings / self._mean_usage_counts


class GroupVectorQuantizer(VanillaVectorQuantizer):
    """
    Group vector quantizer from the paper https://www.isca-speech.org/archive_v0/Interspeech_2019/pdfs/1198.pdf

    Args:
        embedding_dim (int): Dimensionality of each codeword
        num_groups (int): Number of groups in the codebook
        num_embeddings_per_group (int): Number of codewords per group
    """

    def __init__(self, embedding_dim: int, num_groups: int, num_embeddings_per_group: int, ema_gamma: float = 0.99,
                 random_restarts_freq: int = 64):
        super(GroupVectorQuantizer, self).__init__(embedding_dim, num_groups * num_embeddings_per_group, ema_gamma,
                                                   random_restarts_freq)
        self.D = embedding_dim
        self.K = num_groups
        self.M = num_embeddings_per_group
        self.N = self.M * self.K
        self._ones_for_counting = nn.Parameter(torch.ones(self.K, self.M), requires_grad=False)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        self._random_restarts(encodings)
        embedding_distances = self._get_codeword_distances(encodings)
        group_embedding_distances = embedding_distances.view(encodings.size(0), self.K, self.M)
        group_embedding_avg_distances = torch.mean(group_embedding_distances, dim=-1)
        nearest_groups = torch.argmin(group_embedding_avg_distances, dim=-1)
        nearest_group_embeddings = self._get_nearest_group_embeddings(nearest_groups)
        nearest_group_embedding_weights = self._get_nearest_group_embedding_weights(nearest_groups,
                                                                                    group_embedding_distances)
        embeddings = torch.sum(nearest_group_embedding_weights * nearest_group_embeddings, dim=-1)
        self._update_stats(encodings, nearest_groups)
        self._ema_update()
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

    def _update_stats(self, encodings: torch.Tensor, nearest_groups: torch.Tensor):
        if self.training:
            self._train_step += 1
            nearest_groups_one_hot = nn.functional.one_hot(nearest_groups, num_classes=self.K).type(torch.float32)
            batch_group_usage_counts = torch.sum(nearest_groups_one_hot, dim=0).unsqueeze(-1)
            batch_usage_counts = (self._ones_for_counting * batch_group_usage_counts).view(self.N)
            self._total_usage_counts.data += batch_usage_counts
            self._mean_usage_counts.data = self._ema_gamma * self._mean_usage_counts + (1 - self._ema_gamma) * batch_usage_counts
            batch_group_mean_encodings = encodings.T @ nearest_groups_one_hot
            batch_mean_encodings = batch_group_mean_encodings.repeat_interleave(self.M, dim=1)
            self._mean_encodings.data = self._ema_gamma * self._mean_encodings + (1 - self._ema_gamma) * batch_mean_encodings


class AttentionVectorQuantizer(VectorQuantizer):

    def __init__(self, embedding_dim: int, num_embeddings: int, heads: int):
        super(AttentionVectorQuantizer, self).__init__(embedding_dim, num_embeddings)
        assert embedding_dim % heads == 0, f"embedding_dim={embedding_dim} is not divisible by heads={heads}"
        self.head_dim = embedding_dim // heads
        self.heads = heads
        self.linear_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.linear_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.linear_v = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.linear_out = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.apply(self._init_params)

    @staticmethod
    def _init_params(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal(module.weight)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        q, k, v = encodings, self.codebook.T, self.codebook.T
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q, k, v = q.view(-1, self.heads, self.head_dim), k.view(self.K, self.heads, self.head_dim), \
            v.view(self.K, self.heads, self.head_dim)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)  # for broadcasting purposes
        attention_coefficients = torch.matmul(q, k.transpose(-1, -2)) / (torch.tensor(self.D) ** (1/2))
        attention_weights = torch.softmax(attention_coefficients, dim=-1)
        multi_heads = torch.matmul(attention_weights, v).transpose(0, 1).contiguous().view(-1, self.D)
        return self.linear_out(multi_heads)
