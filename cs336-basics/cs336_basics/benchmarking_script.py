
from __future__ import annotations

import sys
import timeit
from pathlib import Path
import argparse

import nvtx
import torch
from einops import einsum

# 仅保留测试所需的模块，移除无用的导入
from model import BasicsTransformerLM as BasicsTransformerLM
from nn_utils import *
from optimizer import *
from data import *


# 这是你复制过来并加了标注的新函数
@nvtx.range("缩放点积注意力 (总体)")
def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    # 假设这是你原来的代码逻辑，现在只是加上了 with 语句

    with nvtx.range("计算注意力分数 (Q * K^T)"):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        with nvtx.range("应用掩码 (Masking)"):
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("计算 Softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("最终矩阵乘法 (Scores * V)"):
        return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
BasicsTransformerLM.scaled_dot_product_attention = annotated_scaled_dot_product_attention
# 保证 tests 可导入
if __name__ == "__main__" and __file__:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

def sync_device(device: str):
    """辅助函数：仅在设备为 CUDA 时进行同步"""
    if "cuda" in device:
        torch.cuda.synchronize()

def main():
    p = argparse.ArgumentParser(description="Benchmark Transformer LM")

    # 实验与路径
    p.add_argument("--run_name", type=str, default="benchmark", help="实验名称")

    # 模型
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--num_layers", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=12)
    p.add_argument("--d_ff", type=int, default=3072)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # 基准测试相关
    p.add_argument("--warmup_steps", type=int, default=10, help="热身步数")
    p.add_argument("--benchmark_steps", type=int, default=50, help="测试步数")
    p.add_argument("--forward_only", action="store_true", help="加上此参数则仅测试正向传播")

    # 训练/优化器参数 (为 AdamW 提供必要参数)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr_max", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    torch.manual_seed(args.seed)

    print(f"========== 开始基准测试 (Device: {args.device}) ==========")
    print \
        (f"Model config: layers={args.num_layers}, d_model={args.d_model}, context={args.context_length}, batch_size={args.batch_size}")
    print(f"Mode: {'Forward Only' if args.forward_only else 'Forward + Backward'}")

    # 1. 初始化模型与优化器
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

    # 2. 生成随机数据 (修复解包问题)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)

    # 3. 阶段一：热身循环 (Warmup)
    print(f"Running warmup for {args.warmup_steps} steps...")
    model.train()
    for _ in range(args.warmup_steps):
        logits = model(x)
        if not args.forward_only:
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # 4. 阶段二：精准计时与基准测试循环
    print(f"Running benchmark for {args.benchmark_steps} steps...")
    sync_device(args.device)  # 清空热身阶段的异步操作
    start_time = timeit.default_timer()

    for _ in range(args.benchmark_steps):
        logits = model(x)
        if not args.forward_only:
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()  # 包含优化器 step 以反映完整的训练单步开销

        sync_device(args.device)  # 强制 CPU 等待当前步完成

    end_time = timeit.default_timer()
    total_time = end_time - start_time

    # 5. 计算并输出各项时间指标
    time_per_step_ms = (total_time / args.benchmark_steps) * 1000
    tokens_per_step = args.batch_size * args.context_length
    tokens_per_sec = tokens_per_step / (total_time / args.benchmark_steps)

    print("\n========== 基准测试结果 ==========")
    print(f"总耗时 (Total Time):          {total_time:.4f} 秒")
    print(f"每步平均耗时 (Time per Step): {time_per_step_ms:.2f} 毫秒")
    print(f"吞吐量 (Throughput):          {tokens_per_sec:.2f} Tokens/秒")
    print("==================================\n")
    print("==================================\n")


if __name__ == "__main__":
    main()