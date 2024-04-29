import torch
import triton
import triton.language as tl

# pylint: disable=W0223,R0913,R0914,W0221

@triton.jit
def _rmsnorm_fwd_fused(
    x_ptr,
    w_ptr,
    output_ptr,
    stride_x_batch,
    stride_x_m,
    stride_x_k,
    stride_rms_w,
    stride_out_batch,
    stride_out_m,
    stride_out_k,
    N: tl.constexpr, # n columns in x
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)

    rows = pid_batch * stride_x_batch + pid_m * stride_x_m
    var = tl.zeros((BLOCK_SIZE,), tl.float32)

    block_N = tl.arange(0, BLOCK_SIZE)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + block_N
        a = tl.load(x_ptr + rows + cols * stride_x_k, mask=cols < N, other=0.).to(tl.float32)
        var += tl.math.pow(a,2)

    var = tl.sum(var, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + block_N
        rms_w = tl.load(w_ptr + cols * stride_rms_w, mask=cols < N)
        x = tl.load(x_ptr + rows + cols * stride_x_k, mask=cols < N, other=0.0).to(tl.float32)

        out = (x * rstd) * rms_w
        out_off = pid_batch * stride_out_batch + pid_m * stride_out_m + cols * stride_out_k
        tl.store(output_ptr + out_off, out, mask=cols < N)


class RMSNormFused(torch.autograd.Function):
    """Fused RMSNorm kernel wrapper"""
    @staticmethod
    def forward(ctx, x, weight, eps):
        y = torch.empty_like(x)
        batch, M, N = x.shape

        MAX_FUSED_SIZE = 65335 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        _rmsnorm_fwd_fused[(batch,M)](
            x, weight, y,
            *x.stride(),
            *weight.stride(),
            *y.stride(),
            N=N, eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y
