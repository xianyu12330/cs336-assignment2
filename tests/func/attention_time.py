import torch
from typing import Callable

def cuda_sync():
    torch.cuda.synchronize()

#创建因果掩码
def causal_mask(seq_len:int,device:torch.device)->torch.Tensor:
    #保留矩阵的下三角部分（包括对角线），其余部分设为0，防止预测偷看后续答案
    return torch.tril(torch.ones((seq_len,seq_len),device=device,dtype=torch.bool))

# fn 是一个可调用对象（函数/模型），接收三个张量参数（q, k, v）,返回一个张量
def time_forward(
        fn: Callable[[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor],
        q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,
        iters:int
)->float:
    torch.cuda.Event
    # 是CUDA事件，用于精确测量GPU操作时间
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # 强制CPU等待GPU完成所有未完成的操作
    cuda_sync()
    start.record()
    for _ in range(iters):
        cuda_sync()
        _ = fn(q,k,v)# 执行前向传播
        cuda_sync()# 5. 每次迭代后同步
    end.record()
    cuda_sync()
    return start.elapsed_time(end) / iters

def time_backward(
        fn: Callable[[torch.Tensor,torch.Tensor,torch.Tensor],torch.Tensor],
        q:torch.Tensor,
        k:torch.Tensor,
        v:torch.Tensor,
        iters:int
)->float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_ms = 0.0
    for _ in  range(iters):
        cuda_sync()
        out = fn(q,k,v)
        loss = out.sum()# 创建标量损失（梯度需要标量）
        cuda_sync()

        start.record()
        loss.backward()
        end.record()
        cuda_sync()
        total_ms += start.elapsed_time(end)
        # 清理梯度（为下一次迭代准备）
        q.grad = None
        k.grad = None
        v.grad = None
    return total_ms / iters

