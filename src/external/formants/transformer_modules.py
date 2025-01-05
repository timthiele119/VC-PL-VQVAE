import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, embed_size, heads):
        
        super(MultiHeadSelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    
    def forward(self, values, keys, query, mask):
        
        # Get number of training examples
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, heads_dim),
        # keys shape: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
    
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, value, key, query, mask):
        
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out


class Encoder(nn.Module):
    
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.scaling = nn.Linear(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask, device):
        
        x = x.to(device)
        mask = mask.to(device)
        
        if x.dim() > 2:
            if x.shape[-1] != 80:
                x = torch.permute(x, dims=(0, 2, 1))
            N, seq_length, feat_dim = x.shape
        else:
            N, seq_length = x.shape
        
        
        positions = torch.arange(0, seq_length).expand(N, seq_length)
        positions = torch.Tensor(positions).to(device)
        scale = self.scaling(x)
        pos_emb = self.position_embedding(positions)
        out = self.dropout(scale + pos_emb)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = MultiHeadSelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, value, key, src_mask, trg_mask, device):
        trg_mask = trg_mask.to(device)
        attention = self.attention(x, x, x, trg_mask) # Masked MultiHeadSelfAttention
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        max_length
    ):
        super(Decoder, self).__init__()
        
        self.scaling = nn.Linear(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, enc_out, src_mask, trg_mask, device):
        
        if x.dim() > 2:
            N, seq_length, feat_dim = x.shape
        else:
            N, seq_length = x.shape
        
        positions = torch.arange(0, seq_length).expand(N, seq_length)
        positions = torch.Tensor(positions).to(device)
        x = self.dropout(
            (self.scaling(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask, device)

        out = self.fc_out(x)

        return out
    
    
class RegressionHead(nn.Module):
    """
    Simple 2-layer downstream head intended for regression purposes (on each time frame).
    """
    
    def __init__(self,
                 emb_dim,
                 hidden_size, 
                 no_outputs):
        """
        Parameters:
            hidden_size: dimensionality of the hidden layers
            no_outputs: number of output neurons
        """
        super(RegressionHead, self).__init__()
        
        self.fc1 = nn.Linear(emb_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, no_outputs)
        
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
    
class Patch_BackProjection(nn.Module):
    
    def __init__(
        self,
        embed_dim,
        orig_in_chans,
        patch_size,
        stride
    ):
        super(Patch_BackProjection, self).__init__()
        output_padding = None
        self.proj = nn.ConvTranspose2d(
            embed_dim, 
            orig_in_chans,
            kernel_size=patch_size, 
            stride=stride
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchDropout(torch.nn.Module):
    """ 
    Implements PatchDropout: https://arxiv.org/abs/2208.07220
    """
    def __init__(
        self, 
        keep_rate=0.7, 
        sampling="uniform", 
        token_shuffling=False, 
        cls_token=False
    ):
        super().__init__()
        assert 0 < keep_rate <=1, "The keep_rate must be in (0,1]"
        
        self.keep_rate = keep_rate
        self.sampling = sampling
        self.token_shuffling = token_shuffling
        self.cls_token = cls_token

    def forward(self, x, force_drop=False):
        """
        If force drop is true it will drop the tokens also during inference.
        """
        if not self.training and not force_drop: return x        
        if self.keep_rate == 1: return x

        # batch, length, dim
        N, L, D = x.shape
        
        # generating patch mask
        patch_mask = self.get_mask(x)
        
        if self.cls_token:
            # making cls mask (assumes that CLS is always the 1st element)
            cls_mask = torch.zeros(N, 1, dtype=torch.int64, device=x.device)
            # cat cls and patch mask
            patch_mask = torch.hstack([cls_mask, patch_mask])
        
        # gather tokens
        x = torch.gather(x, dim=1, index=patch_mask.unsqueeze(-1).repeat(1, 1, D))

        return x
    
    
    def get_mask(self, x):
        
        if self.sampling == "uniform":
            return self.uniform_mask(x)
        else:
            return NotImplementedError(f"PatchDropout does not support {self.sampling} sampling")
    
    
    def uniform_mask(self, x):
        """
        Returns an id-mask using uniform sampling
        """
        N, L, D = x.shape
        _L = L - 1 if self.cls_token else L # patch length (without CLS)
        
        keep = int(_L * self.keep_rate)
        patch_mask = torch.rand(N, _L, device=x.device)
        patch_mask = torch.argsort(patch_mask, dim=1) + 1
        patch_mask = patch_mask[:, :keep]
        if not self.token_shuffling:
            patch_mask = patch_mask.sort(1)[0]
        
        return patch_mask