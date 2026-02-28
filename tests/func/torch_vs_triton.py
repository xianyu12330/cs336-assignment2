import torch
import math
import pandas as pd
import triton
import triton.testing
from tests.func.flash_attn import FlashAttention2Triton,FlashAttention2_PyTorch

def run_benchmark():
    BATCH_SIZE = 1
    SEQ_LEN = [128,256,512,1024,2048,4096,8192,16384,32768,65536]
    HEAD_DIMS = [16,32,64,128]
    DTYPE = [torch.bfloat16,torch.float32]
    pytorch_att = FlashAttention2_PyTorch()
    triton_att = FlashAttention2Triton()
    result = []
    print("开始执行基准测试...")
    #遍历所有组合
    for dtype in DTYPE:
        for dim in HEAD_DIMS:
            for seq_len in SEQ_LEN:
                print(f"Testing: SeqLen={seq_len}, Dim={dim}, Dtype={dtype}")
                # 每次测试前清空缓存，防止前一次测试的内存碎片导致本次误报 OOM
                torch.cuda.empty_cache()
                # --- 1. 数据准备 ---
                # 开启 requires_grad=True 以便测试反向传播
                q = torch.randn(BATCH_SIZE, SEQ_LEN, dim,device='cuda',dtype=dtype,requires_grad=True)
                k = torch.randn(BATCH_SIZE, SEQ_LEN, dim, device='cuda',dtype=dtype,requires_grad=True)
                v = torch.randn(BATCH_SIZE, SEQ_LEN, dim, device='cuda',dtype=dtype,requires_grad=True)
                #模拟上游传来的梯度
                dO = torch.randn_like(q)
                row = {
                    "Seq_len": seq_len,
                    "Dim": dim,
                    "Dtype": str(dtype).split('.')[-1]
                }
                #-------2------包装测试函数
                #【pytorch】版本
                def pytorch_fwd():
                    return pytorch_att.forward(q,k,v,is_causal=True)

                def pytorch_bwd():
                    out = pytorch_att.forward(q,k,v,is_causal=True)
                    out.backward(dO,retain_graph=True)

                def pytorch_e2e():
                    q.grad,k.grad,v.grad = None,None,None
                    out = pytorch_att.forward(q,k,v,is_causal=True)
                    out.backward(dO,retain_graph=True)

                #[triton]版本
                def triton_fwd():
                    return triton_att.forward(q,k,v,is_causal=True)

                def triton_bwd():
                    out = triton_att.forward(q,k,v,is_causal=True)
                    out.backward(dO,retain_graph=True)

                def triton_e2e():
                    q.grad,k.grad,v.grad = None,None,None
                    out = triton_att.forward(q,k,v,is_causal=True)
                    out.backward(dO,retain_graph=True)

                #----3.执行基准测试并捕捉 OOM ---
                def bench_with_oom_handling(fn):
                    try:
                        # do_bench 返回的是一个以毫秒为单位的时间 (默认是中位数)
                        ms = triton.testing.do_bench(fn,quantiles = None)
                        return f"{ms:.3f}ms"
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            return "OOM"
                        else:
                            raise e  # 抛出其他非 OOM 错误
                #开始测试时间
                row['pytorch_fwd(ms)'] = bench_with_oom_handling(pytorch_fwd)
                row['triton_fwd(ms)'] = bench_with_oom_handling(triton_fwd)

                row['pytorch_bwd(ms)'] = bench_with_oom_handling(pytorch_bwd)
                row['triton_bwd(ms)'] = bench_with_oom_handling(triton_bwd)

                row['pytorch_e2e(ms) '] = bench_with_oom_handling(pytorch_e2e)
                row['triton_e2e(ms) '] = bench_with_oom_handling(triton_e2e)
                result.append(row)

    #4.汇总输出
    df = pd.DataFrame(result)
    print("\n============基准结果===========")
    print(df.to_markdown(index=False))
    df.to_csv("flash_attention_benchmark.csv", index=False)  # 也可以保存为 csv 提交
if __name__ == '__main__':
    run_benchmark()