import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False

from .bitlinear import BitLinear
from .utils import apply_rotary_emb, precompute_freqs_cis, RMSNorm
from .config import BoBConfig
from .moe import MoeLayer

# pylint: disable=W0223,R0402,R0902
class EagerAttention(nn.Module):
    """Attention Module with BitLinear weights"""
    def __init__(self, config: BoBConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_dim // config.num_heads
        self.n_heads = config.num_heads
        self.hidden_size = config.hidden_dim
        self.attn_dropout = config.attn_dropout
        self.proj_dropout = config.proj_dropout
        self.memory_efficient = config.memory_efficient_attention

        self.wq = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.wk = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.wv = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.attn_drop = nn.Dropout(self.attn_dropout)
        self.wo = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.proj_drop = nn.Dropout(self.proj_dropout)

    def forward(
            self,
            x: torch.Tensor,
            freq_cis: torch.Tensor,
            mask: torch.Tensor = None,
            inference: bool = False,
        ):
        # pylint: disable=W0511

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x, inference), self.wk(x, inference), self.wv(x, inference)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freq_cis)

        keys = xk.transpose(1,2)
        values = xv.transpose(1,2)
        queries = xq.transpose(1,2)

        # TODO: flashattn2
        attn = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn + mask[:,:, :seqlen, :seqlen]
        attn = F.softmax(attn.float(), dim=-1).type_as(queries)
        attn_output = torch.matmul(attn, values)

        attn_output = self.attn_drop(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(attn_output, inference)
    
class FlashAttention(EagerAttention):
    """FlashAttention Module with BitLinear weights"""
    def __init__(
            self,
            config: BoBConfig,
        ):
        super().__init__(config)
    
    def forward(
            self,
            x: torch.Tensor,
            freq_cis: torch.Tensor,
            mask: torch.Tensor = None,
            inference: bool = False,
        ):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x, inference), self.wk(x, inference), self.wv(x, inference)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freq_cis)

        keys = xk.transpose(1,2)
        values = xv.transpose(1,2)
        queries = xq.transpose(1,2)

        attn_output = flash_attn_func(
            queries,
            keys,
            values,
            dropout_p=self.attn_drop,
            softmax_scale=None, 
            causal=False,
        )

        attn_output = self.attn_drop(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(attn_output, inference)

class FeedForward(nn.Module):
    """FeedForward SwiGLU module with BitLinear weights"""
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
        ):
        super().__init__()
        self.w1 = BitLinear(dim, hidden_dim)
        self.w2 = BitLinear(hidden_dim, dim)
        self.w3 = BitLinear(dim, hidden_dim)

    def forward(self, x, inference: bool = False):
        return self.w2(F.silu(self.w1(x, inference)) * self.w3(x, inference), inference=inference)

class TransformerBlock(nn.Module):
    """Transformer MoE based block"""
    def __init__(
            self,
            config: BoBConfig,
        ):
        super().__init__()
        if config.memory_efficient_attention and FLASH_ATTN_AVAILABLE:
            self.attn = FlashAttention(config)
        else:
            logging.warning("Using Eager Attention. Consider using memory_efficient_attention=True.")
            self.attn = EagerAttention(config)
        
        if config.bitlinear_gates:
            gate = BitLinear(config.hidden_dim, config.num_experts)
        else:
            gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        self.ff = MoeLayer(
            experts=[
                FeedForward(
                    config.hidden_dim,
                    config.hidden_dim
                ) for _ in range(config.num_experts)
                ],
            gate=gate,
            num_experts=config.num_experts,
            num_experts_per_token=config.num_experts_per_token,
        )

    def forward(
            self,
            x: torch.Tensor,
            freq_cis: torch.Tensor,
            inference: bool = False,
        ):
        # bitlinear has built-in rmsnorm
        h = x + self.attn(x, freq_cis, mask=None, inference=inference)
        return h + self.ff(h, inference=inference)

class Transformer(nn.Module):
    """
    Transformer module based on the Mixtral architecture.

    Args:
        config (bixtralConfig): Configuration object containing model hyperparameters.

    Attributes:
        config (bixtralConfig): Configuration object containing model hyperparameters.
        embed (nn.Embedding): Embedding layer for input tokens.
        blocks (nn.ModuleList): List of TransformerBlock instances.
        norm (RMSNorm): Normalization layer.
        freq_cis (torch.Tensor): Precomputed frequency tensor.
        vocab_proj (nn.Linear): Linear projection layer for output vocabulary.
        mask (torch.Tensor): Mask for attention mechanism.
    """

    def __init__(
            self,
            config: BoBConfig,
        ):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList([
                TransformerBlock(config) for _ in range(config.num_layers)
            ])
        self.norm = RMSNorm(config.hidden_dim)
        self.freq_cis = precompute_freqs_cis(
            config.hidden_dim // config.num_heads, config.seq_len * 2
        )
        self.vocab_proj = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if not config.memory_efficient_attention:
            mask = torch.full(
                (1,1,config.seq_len,config.seq_len),
                float('-inf'),
                device=self.freq_cis.device,
                dtype=self.freq_cis.dtype
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer('mask', mask)
        else:
            mask = None

    def forward(
            self,
            x: torch.Tensor,
            inference: bool = False,
        ):
        _, seqlen = x.shape
        x = self.embed(x)
        freq_cis = self.freq_cis[:seqlen].to(x.device)
        for _, blk in enumerate(self.blocks):
            x = blk(x, freq_cis, inference=inference)
        x = self.norm(x)
        return self.vocab_proj(x)
