from bob import RMSNorm
from bob.kernels import RMSNormFused
import torch

rms_norm = RMSNormFused.apply

def test_rmsnorm_on_cuda():
    assert torch.cuda.is_available(), "cuda device not available for testing rmsnorm"
    batch = 1
    M = 1151
    N = 8192
    eps = 1e-5
    dtype = torch.float32
    
    x_shape = (batch,M,N)
    w_shape = (x_shape[-1], )
    weight = torch.ones(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    x.requires_grad_(True)
    eager_rmsnorm = RMSNorm(
        x_shape[-1],
        eps=eps
    ).to('cuda')
    # forward pass
    y_tri = rms_norm(x, weight, eps)
    y_ref = eager_rmsnorm(x).to(dtype)

    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0), f"{y_tri} != {y_ref}"