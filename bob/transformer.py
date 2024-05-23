from typing import Optional
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import BitLinear
from .utils import RotaryEmbeddings, RMSNorm, apply_rotary_pos_emb, repeat_kv
from .config import BoBConfig
from .moe import MoeLayer

# TODO: add support for FlashAttention from Dao
FLASH_ATTN_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')

# pylint: disable=W0223,R0402,R0902
class EagerAttention(nn.Module):
    """Attention Module with BitLinear weights"""
    def __init__(self, config: BoBConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_dim // config.num_heads
        self.n_heads = config.num_heads
        self.n_kv_heads = config.num_kv_heads
        self.kv_groups = config.num_heads // config.num_kv_heads
        self.hidden_size = config.hidden_dim
        self.attn_dropout = config.attn_dropout
        self.proj_dropout = config.proj_dropout

        self.rotary_emb = RotaryEmbeddings(
            self.head_dim,
            max_seq_len=config.seq_len,
            theta=config.rope_theta,
        )
        self.wq = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.wk = BitLinear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = BitLinear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.attn_drop = nn.Dropout(self.attn_dropout)
        self.wo = BitLinear(self.hidden_size, self.hidden_size, bias=False)
        self.proj_drop = nn.Dropout(self.proj_dropout)

    def forward(
            self,
            x: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,
            mask: Optional[torch.Tensor] = None,
            inference: Optional[bool] = False,
        ):
        # pylint: disable=W0511

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x, inference), self.wk(x, inference), self.wv(x, inference)
        queries = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1,2)
        keys = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1,2)
        values = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1,2)

        cos, sin = self.rotary_emb(values, seq_len=seqlen)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, position_ids=position_ids)
        
        keys = repeat_kv(keys, self.kv_groups)
        values = repeat_kv(values, self.kv_groups)

        attn = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn + mask[:,:, :seqlen, :seqlen]
        attn = F.softmax(attn.float(), dim=-1).type_as(queries)
        attn_output = torch.matmul(attn, values)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        attn_output = self.wo(attn_output, inference)
        return attn_output
    
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
            position_ids: Optional[torch.LongTensor] = None,
            mask: Optional[torch.Tensor] = None,
            inference: Optional[bool] = False,
        ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x, inference), self.wk(x, inference), self.wv(x, inference)
        queries = xq.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1,2)
        keys = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1,2)
        values = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1,2)

        cos, sin = self.rotary_emb(values, seq_len=seqlen)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin, position_ids=position_ids)

        keys = repeat_kv(keys, self.kv_groups)
        values = repeat_kv(values, self.kv_groups)

        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=self.training,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        attn_output = self.wo(attn_output, inference)
        return attn_output

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
                    config.intermediate_dim,
                ) for _ in range(config.num_experts)
                ],
            gate=gate,
            num_experts=config.num_experts,
            num_experts_per_token=config.num_experts_per_token,
        )

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inference: bool = False,
        ) -> torch.Tensor:
        # bitlinear has built-in rmsnorm
        h = self.attn(
            x,
            mask=mask,
            position_ids=position_ids,
            inference=inference,
            )
        h = x + h
        return h + self.ff(h, inference=inference)

class Transformer(nn.Module):
    """
    Transformer module based on SMoE architecture.

    Args:
        config (BoBConfig): Configuration object containing model hyperparameters.
    """

    def __init__(
            self,
            config: BoBConfig,
        ):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.blocks = nn.ModuleList([
                TransformerBlock(config, layer_idx) for layer_idx in range(config.num_layers)
            ])
        self.norm = RMSNorm(config.hidden_dim)
        self.vocab_proj = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        if not config.memory_efficient_attention:
            mask = torch.full(
                (1,1,config.seq_len,config.seq_len),
                float('-inf'),
            )
            mask = torch.triu(mask, diagonal=1)
        else:
            mask = None
        self.register_buffer('mask', mask)

    def forward(
            self,
            x: torch.Tensor,
            position_ids: Optional[torch.LongTensor] = None,
            inference: bool = False,
        ):
        _, seqlen = x.shape
        if position_ids is None:
            device = x.device
            position_ids = torch.arange(
                0, seqlen, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seqlen)
        else:
            position_ids = position_ids.view(-1, seqlen).long()

        x = self.embed(x)
        for _, blk in enumerate(self.blocks):
            x = blk(
                x,
                mask=self.mask,
                position_ids=position_ids,
                inference=inference
            )
        x = self.norm(x)
        return self.vocab_proj(x)
