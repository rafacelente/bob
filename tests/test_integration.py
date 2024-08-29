
def test_transformer_forward():
    from bob import Transformer, BoBConfig
    import torch

    debug_config = BoBConfig(
        vocab_size=2000,
        seq_len=64,
        rope_theta=1000.0,
        hidden_dim=256,
        intermediate_dim=512,
        num_heads=2,
        num_kv_heads=1,
        num_layers=2,
        num_experts=2,
        num_experts_per_token=1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        memory_efficient_attention=True,
    )

    transformer = Transformer(debug_config)
    
    x = torch.randint(0,debug_config.vocab_size, (2,debug_config.seq_len))
    output, _ = transformer(x)

    assert output.shape == (2, debug_config.seq_len, debug_config.vocab_size)

def test_transformer_backward():
    from bob import Transformer, BoBConfig
    import torch

    debug_config = BoBConfig(
        vocab_size=2000,
        seq_len=64,
        rope_theta=1000.0,
        hidden_dim=256,
        intermediate_dim=512,
        num_heads=2,
        num_kv_heads=1,
        num_layers=2,
        num_experts=2,
        num_experts_per_token=1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        memory_efficient_attention=True,
    )

    transformer = Transformer(debug_config)
    
    x = torch.randint(0,debug_config.vocab_size, (2,debug_config.seq_len))
    y = torch.randint(0,debug_config.vocab_size, (2,debug_config.seq_len))
    output, _ = transformer(x)
    loss = torch.nn.CrossEntropyLoss()(output.view(-1,debug_config.vocab_size), y.view(-1))
    loss.backward()
    assert True

def test_transformer_quantized_inference_with_kv_cache():
    from bob import Transformer, BoBConfig, BitLinear, quantize_weights_to_int8
    import torch

    debug_config = BoBConfig(
        vocab_size=2000,
        seq_len=64,
        rope_theta=1000.0,
        hidden_dim=256,
        intermediate_dim=512,
        num_heads=2,
        num_kv_heads=1,
        num_layers=2,
        num_experts=2,
        num_experts_per_token=1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        memory_efficient_attention=True,
    )

    transformer = Transformer(debug_config)

    # quantize all bitlinear layers
    for _, layer in transformer.named_modules():
            if isinstance(layer, BitLinear):
                for k, v in layer.state_dict().items():
                    if 'weight' in k and 'norm' not in k:
                        w_quant, scale = quantize_weights_to_int8(v)
                        layer.weight.requires_grad = False
                        layer.weight.data = w_quant
                        layer.weight_scale = scale
    
    x = torch.randint(0,debug_config.vocab_size, (1,debug_config.seq_len))
    
    with torch.no_grad():
        output, _ = transformer(x, inference=True, use_cache=True)

    assert output.shape == (1, debug_config.seq_len, debug_config.vocab_size)

def test_transformer_quantized_inference():
    from bob import Transformer, BoBConfig, BitLinear, quantize_weights_to_int8
    import torch

    debug_config = BoBConfig(
        vocab_size=2000,
        seq_len=64,
        rope_theta=1000.0,
        hidden_dim=256,
        intermediate_dim=512,
        num_heads=2,
        num_kv_heads=1,
        num_layers=2,
        num_experts=2,
        num_experts_per_token=1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        memory_efficient_attention=True,
    )

    transformer = Transformer(debug_config)

    # quantize all bitlinear layers
    for _, layer in transformer.named_modules():
            if isinstance(layer, BitLinear):
                for k, v in layer.state_dict().items():
                    if 'weight' in k and 'norm' not in k:
                        w_quant, scale = quantize_weights_to_int8(v)
                        layer.weight.requires_grad = False
                        layer.weight.data = w_quant
                        layer.weight_scale = scale
    
    x = torch.randint(0,debug_config.vocab_size, (1,debug_config.seq_len))
    
    with torch.no_grad():
        output, _ = transformer(x, inference=True)

    assert output.shape == (1, debug_config.seq_len, debug_config.vocab_size)