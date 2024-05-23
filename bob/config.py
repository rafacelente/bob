from dataclasses import dataclass

# pylint: disable=C0115,R0902
@dataclass
class BoBConfig:
    vocab_size: int = 32000
    seq_len: int = 1024
    rope_theta: float = 10000.0
    hidden_dim: int = 2048 # 4096
    intermediate_dim: int = 4096 # 14336
    num_heads: int = 8 #32
    num_kv_heads: int = 2 # 8
    num_layers: int = 12 # 32
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    bias: bool = False
    memory_efficient_attention: bool = True
    bitlinear_gates: bool = False
    num_experts: int = 4
    num_experts_per_token: int = 2
    # head_dim = hidden_dim // num_heads  = 128

@dataclass
class TrainerConfig:
    max_steps: int = 10000
    gpus: int = 1
    precision: int = 16
    gradient_clip_val: float = 1.0
    check_val_every_n_epoch: int = 1
    log_every_n_steps: int = 1
    limit_val_batches: float = 1.0
    limit_train_batches: float = 1.0
    num_warmup_steps: int = 1000
