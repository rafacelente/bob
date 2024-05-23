from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from .utils import RMSNorm
from .quantization import weight_quant, activation_quant, activation_post_quant, quantize_weights_to_int8

class BitLinear(nn.Linear):
    """
        Quantized-aware linear layer that casts 1.58-bit weights.
        Based on https://arxiv.org/abs/2402.17764.

        Args:
            in_features (int): size of each input sample
            out_features (int): size of each output sample
            bias (bool): If set to False, the layer will not learn an additive bias. Default: False

        Attributes:
            rms_norm (RMSNorm): RMSNorm layer
            weight_scale (torch.Tensor): scale factor for weights for inference.
                                            This applied on post-training quantization.
    """
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.rms_norm = RMSNorm(in_features)
        self.weight_scale = None
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def quantize_(self):
        """
            Quantize the weights to 1.58-bit
            and create weight_scale for inference.
        """
        for k, v in self.state_dict().items():
            if 'weight' in k and 'norm' not in k:
                w_quant, scale = quantize_weights_to_int8(v)
                self.weight.requires_grad = False
                self.weight.data = w_quant
                self.weight_scale = scale


    def forward(self, x, inference: Optional[bool]=False):
        w = self.weight
        x_norm = self.rms_norm(x)
        if not inference:
            x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
            w_quant = w + (weight_quant(w) - w).detach()
            return F.linear(x_quant, w_quant, self.bias)
        # in case of inference, the weights are
        # offline quantized to int8, so we assume w = w_quant
        x_quant, x_scale = activation_post_quant(x_norm)
        w_scale = self.weight_scale
        # to be replaced by a low bit gemm kernel
        return F.linear(x_quant, w.float(), self.bias) / (x_scale * w_scale)
