import pytest

def test_eager_and_flash_attn_equal_outputs():
    from bob import EagerAttention, FlashAttention, BoBConfig
    import torch

    eager_config = BoBConfig(
        dim=128,
        head_dim=16,
        hidden_dim=256,
        num_heads=4,
        num_kv_heads=4,
        num_layers=4,
        attn_dropout=0.1,
        memory_efficient_attention=False,
    )

    flash_config = BoBConfig(
        dim=128,
        head_dim=16,
        hidden_dim=256,
        num_heads=4,
        num_kv_heads=4,
        num_layers=4,
        attn_dropout=0.1,
        memory_efficient_attention=True,
    )

    eager_attn = EagerAttention(eager_config)
    flash_attn = FlashAttention(flash_config)
    x = torch.randn(2, 16, 128)
    freq_cis = torch.randn(2, 16, 128)
    eager_output = eager_attn(x, freq_cis)
    flash_output = flash_attn(x, freq_cis)

    assert torch.allclose(eager_output, flash_output, atol=1e-5)