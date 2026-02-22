import time
from collections.abc import Callable
from statistics import mean


import torch
from torch import nn
from torch.version import cuda
from profile_sleep import *

class MLP(nn.Module):
    """Simple MLP: linear -> GeLU -> linear -> GeLU -> ... -> linear -> GeLU"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])
    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.gelu(x)
        return x
def run_mlp(dim: int, num_layers: int, batch_size: int, num_steps: int) -> Callable:
    # Define a model (with random weights)
    model = MLP(dim, num_layers).to('cuda')
    # Define an input (random)
    x = torch.randn(batch_size, dim, device='cuda')
    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        for step in range(num_steps):
            # Forward
            y = model(x).mean()
            # Backward
            y.backward()
    return run


def benchmark(description: str, run: Callable, num_warmups: int = 1, num_trials: int = 3):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    #warmup阶段
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    # Time it for real now!
    times: list[float] = [] # @inspect times, @inspect description
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = time.time()

        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

        end_time = time.time()
        times.append((end_time - start_time) * 1000) # @inspect times

    mean_time = mean(times) # @inspect mean_time
    print(f"{description}: {mean_time:.3f} ms")
    return mean_time
def run_operation1(dim: int, operation: Callable) -> Callable:
    # Setup: create one random dim x dim matrices
    x = torch.randn(dim, dim, device='cuda')
    # Return a function to perform the operation
    return lambda : operation(x)

def run_operation2(dim: int, operation: Callable) -> Callable:
    # Setup: create two random dim x dim matrices
    x = torch.randn(dim, dim, device='cuda')
    y = torch.randn(dim, dim, device='cuda')
    # Return a function to perform the operation
    return lambda : operation(x, y)
if __name__ == "__main__":
    if cuda_gelu is not None:
        check_equal(cuda_gelu, manual_gelu)

    pytorch_time = benchmark("pytorch_gelu", run_operation1(dim=16384, operation=pytorch_gelu))  # @inspect pytorch_time
    manual_time = benchmark("manual_gelu", run_operation1(dim=16384, operation=manual_gelu))  # @inspect manual_time
    if cuda_gelu is not None:
        cuda_time = benchmark("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu))  # @inspect cuda_time
        cuda_gelu_profile = profile("cuda_gelu", run_operation1(dim=16384, operation=cuda_gelu))

    print(mlp_profile)








