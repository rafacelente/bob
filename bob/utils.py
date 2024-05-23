from typing import Tuple, Optional
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """Eager RMSNorm used by BitLinear"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x

class RotaryEmbeddings(nn.Module):
    """
        Faster (yet still eager) Rotary Embeddings
        Adapted from transformers.src.transformers.models.mixtral.modeling_mixtral
    """
    def __init__(
            self,
            dim: int,
            max_seq_len: int,
            theta=10000.0,
            device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(self.device) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=self.max_seq_len, device=self.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(
            self,
            seq_len: int,
            device: Optional[bool]=None,
            dtype: Optional[torch.dtype]=None,
            ):
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cache', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cache', emb.sin().to(dtype), persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            seq_len: Optional[int]=None,
            ):
        if seq_len is None:
            seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len, self.device, x.dtype)

        return (
            self.cos_cache[:seq_len].to(x.dtype),
            self.sin_cache[:seq_len].to(x.dtype),
        )

def rotate_half(x: torch.Tensor):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        position_ids: torch.Tensor,
    ):
    cos = cos_cache[position_ids]
    sin = sin_cache[position_ids]
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)