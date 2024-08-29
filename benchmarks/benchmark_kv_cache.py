import torch
import torch.nn.functional as F
from typing import List, Optional
import torch.nn as nn
from bob import BoBConfig, Transformer, BitLinear, quantize_weights_to_int8
import time

DEBUG_CONFIG = BoBConfig(
        vocab_size=2000,
        seq_len=512,
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

class Benchmark:
    def __init__(
            self,
            input_example_size: Optional[int] = 10,
            num_test_tokens: Optional[int] = None,
            quantized_inference: Optional[bool] = True
        ):
        assert input_example_size <= DEBUG_CONFIG.seq_len, "Input example size is too large"
        if num_test_tokens is None:
            assert input_example_size < DEBUG_CONFIG.seq_len, "Input example size is bigger than sequence length."
            num_test_tokens = DEBUG_CONFIG.seq_len - input_example_size
        self.num_test_tokens = num_test_tokens
        self.transformer1 = Transformer(DEBUG_CONFIG)
        self.transformer2 = Transformer(DEBUG_CONFIG)
        self.transformer2.load_state_dict(self.transformer1.state_dict())
        if quantized_inference:
            self._quantize_weights_to_ternary(self.transformer1)
            self._quantize_weights_to_ternary(self.transformer2)
        self.quantized_inference = quantized_inference
        self.input_example = torch.randint(
            0,DEBUG_CONFIG.vocab_size, 
            (1, input_example_size))
    
    def _quantize_weights_to_ternary(self, model: nn.Module):
        for _, layer in model.named_modules():
            if isinstance(layer, BitLinear):
                for k, v in layer.state_dict().items():
                    if 'weight' in k and 'norm' not in k:
                        w_quant, scale = quantize_weights_to_int8(v)
                        layer.weight.requires_grad = False
                        layer.weight.data = w_quant
                        layer.weight_scale = scale

    def _inference(
            self, 
            model: nn.Module,
            past_key_value: Optional[List[torch.FloatTensor]] = None,
            x: Optional[torch.Tensor] = None, 
            use_cache: bool = True
        ):
        model.eval()
        with torch.no_grad():
            output, cache = model(
                x,
                past_key_value=past_key_value,
                inference=self.quantized_inference,
                use_cache=use_cache
            )
        return output, cache

    @torch.no_grad()  
    def run_and_time_inference(self, model: nn.Module, use_cache: bool, x: Optional[torch.Tensor] = None):
        start = time.time()
        if x is None:
            x = self.input_example
        past_key_value = None
        # prompt cache
        for i in range(self.num_test_tokens):
            if i == 0 and use_cache:
                output, past_key_value = self._inference(
                    model=model,
                    x=x,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
            else:
                output, past_key_value = self._inference(
                    model=model,
                    x=x if not use_cache else x[:, -1:],
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(F.softmax(next_token_logits, dim=-1), dim=-1, keepdims=True)
            x = torch.cat([x, next_token], dim=-1)
        end = time.time()
        return x, end - start

    def run_benchmark(self, print_results: bool = False):
        print(f"Starting KV Cache benchmark | Input example shape: {self.input_example.shape[-1]} | Num test tokens: {self.num_test_tokens} | Quantized inference: {self.quantized_inference}")
        print("-"*105)
        print("Running inference without cache")
        output1, time1 = self.run_and_time_inference(self.transformer1, use_cache=False)
        print("-"*105)
        print("Running inference with cache")
        output2, time2 = self.run_and_time_inference(self.transformer2, use_cache=True)
        print("-"*105)
        if print_results:
            print(f"Time without cache: {time1}")
            print(f"Time with cache: {time2}")
        print(f"output without cache: {output1}")
        print(f"output with cache: {output2}")
        assert torch.allclose(output1, output2, atol=1e-5), "Outputs are not equal"
        return time1, time2

if __name__ == "__main__":
    benchmark = Benchmark(
        input_example_size=2,
        num_test_tokens=5,
        quantized_inference=False
    )
    benchmark.run_benchmark(print_results=True)