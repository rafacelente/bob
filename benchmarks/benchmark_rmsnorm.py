import torch
import triton
from bob import RMSNorm
from bob.kernels import RMSNormFused

rms_norm = RMSNormFused.apply

def gbps(ms, x: torch.Tensor):
        return 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 101)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=[
            'fused-rmsnorm',
            'eager-rmsnorm',
        ],  # possible values for `line_arg``
        line_names=[
            "Fused RMSNorm",
            "Eager RMSNorm",
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="rmsnorm-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096, 'batch': 1},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(batch, M, N, provider):
    x = torch.randn(batch,M, N, device='cuda', dtype=torch.float32)
    weight = torch.ones(N, dtype=torch.float32, device='cuda', requires_grad=True)
    eps = 1e-5
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'fused-rmsnorm':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rms_norm(x, weight, eps), quantiles=quantiles)
    if provider == 'eager-rmsnorm':
        eager_rmsnorm = RMSNorm(
            N,
            eps=eps
        ).to('cuda')
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: eager_rmsnorm(x), quantiles=quantiles)
    return gbps(ms, x), gbps(max_ms, x), gbps(min_ms, x)


# benchmark.run(show_plots=True, print_data=True)