import math

import torch
import triton
import triton.language as tl


class FlashAttention2_PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k,v, is_causal=False):
        """
        q:[B,S,D],k,v:[B,K,D]
        前向传播定义
        注意：ctx 是 context 对象，用于存储反向传播需要的变量。
        """
        # --- 1. 获取维度并定义 Tile (分块) 大小 ---
        # 截图要求 Tile 大小至少为 16x16，且测试用例都是 2 的幂
        B_r = 16  # Q 的分块大小
        B_c = 16  # K, V 的分块大小

        # 获取输入形状，通常 Q, K, V 的 shape 为 (Batch, Num_Heads, Seq_Len, Head_Dim)
        # 你的实现需要预先分配输出 O 以及用于反向传播的 L (logsumexp) 的显存
        B,S,D = q.shape
        scale = 1.0 / math.sqrt(D)
        O = torch.empty((B,S,D), device=q.device, dtype=q.dtype)
        L = torch.empty((B,S), device=q.device, dtype=q.dtype)
        # ... [这里将是你实现 FlashAttention-2 核心的分块双重循环逻辑] ...
        # 注意：作业明确说明目前可以忽略 is_causal=True 的掩码逻辑
        #从0~seq_len每次增长一块大小b_r
        for i in range(0,S,B_r):
            #取所有行的i:i+B_r列
            q_block = q[:,i:i+B_r,:]
            #累计输出
            O_acc = torch.zeros((B,B_r,D), device=q.device, dtype=torch.float32)
            #累计的 softmax 分母-因为每次都会更新最大值 m之前的 exp 需要重新缩放。
            l = torch.zeros((B,B_r), device=q.device, dtype=torch.float32)
            #当前块累计的最大值,每个 query 位置目前为止看到的 score 最大值
            m = torch.full((B,B_r),-float('inf'), device=q.device, dtype=torch.float32)
            for j in range(0,S,B_c):
                #加载分块的kv到速度快的
                k_j = k[:,j:j+B_c,:]
                v_j = v[:,j:j+B_c,:]
                ## S = q @ k^T * scale -> (B, B_r, B_c)
                pre_att_score = torch.matmul(q_block,k_j.transpose(-1,-2)) * scale
                #softmax update  ->(沿着最后一个维度压缩->(B,B_r)
                m_new = torch.maximum(m,pre_att_score.max(dim=-1).values)
                #exp(S - m_new) ->(B,B_r,B_c)
                p_j = torch.exp(pre_att_score - m_new.unsqueeze(-1))
                # l_new = exp(m - m_new) * l + rowsum(P_tilde)
                #(B,B_r)
                l_new = torch.exp(m - m_new) * l + p_j.sum(dim=-1)
                O_acc = (torch.exp(m - m_new)).unsqueeze(-1) * O_acc + torch.matmul(p_j,v_j)

                m,l = m_new,l_new
            o_i = O_acc / l.unsqueeze(-1)
            O[:,i:i+B_c,:] = o_i.to(q.device)
            # L = logsumexp(S_row) = m + log(l)
            L[:,i:i + B_r] = m + torch.log(l)
        ctx.save_for_backward(L, q, k, v, O)
        ctx.is_causal = is_causal

        # return output
        return O

        # 假设你计算出了最终的输出 O 以及对数指数和 L (logsumexp value)
        # O = ...
        # L = ...

        # --- 2. 保存反向传播所需的上下文 (核心要求) ---
        # 截图要求："use save L, Q, K, V, O for the backward pass"
        ctx.save_for_backward(L, Q, K, V, O)

        # --- 3. 返回输出 ---
        # 截图要求："return O"
        return O

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播定义
        """
        # 截图要求："for now you can make it just raise NotImplementedError"
        # 也就是说第一步你只需要专注前向传播即可
        raise NotImplementedError("FlashAttention-2 的反向传播暂未实现")

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,#指向数据在显存中起始位置的基指针
    O_ptr, L_ptr,#Output输出矩阵和logsumexp辅助矩阵的指针
    stride_qb, #移动到下一个 Batch 需要跳过的距离。
    stride_qq, #移动到下一个 Query 序列位置需要跳过的距离。
    stride_qd,#移动到下一个特征维度需要跳过的距离。
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,# Output的 strides
    stride_lb, stride_lq, # Logsumexp的 strides
    N_QUERIES, N_KEYS,# query,key/value序列的长度
    scale,
    D: tl.constexpr,# 每个头的维度
    Q_TILE_SIZE: tl.constexpr,#Bq
    K_TILE_SIZE: tl.constexpr,#Bk
):
    #当前这段代码负责处理的块的坐标。
    pid_q = tl.program_id(0)
    #表示当前是第几个 Batch
    pid_b = tl.program_id(1)
    #每个 program instance 负责：一个 batch 的一个 Q tile
    #每个 block 计算：Q[pid_b, pid_q_tile, :]
    q_offsets = pid_q * Q_TILE_SIZE + tl.arange(0,Q_TILE_SIZE)
    d_offsets = tl.arange(0,D)

    q_ptrs = Q_ptr \
            + pid_b * stride_qb \
            + q_offsets[:,None] * stride_qq \
            + d_offsets[None:,] * stride_qd
    q = tl.load(q_ptrs,mask=(q_offsets[:,None] < N_QUERIES),other=0.0).to(tl.float32)
    #初始化 Online Softmax 状态
    ## 当前最大值
    m = tl.full((Q_TILE_SIZE,), -float("inf"),tl.float32)#[bq]
    ## 当前 softmax 分母
    l = tl.zeros((Q_TILE_SIZE,),tl.float32)
    ## 当前加权和
    acc = tl.zeros((Q_TILE_SIZE,D),tl.float32)
    #每次处理一个 K block
    for kb in tl.static_range(0,N_KEYS,K_TILE_SIZE):
        k_offsets = kb + tl.arange(0,K_TILE_SIZE)
        k_ptrs = K_ptr + pid_b * stride_kb + k_offsets[:,None] * stride_kk
        v_ptrs = V_ptr + pid_b * stride_vb + k_offsets[:,None] * stride_vk
        k = tl.load(k_ptrs,mask=(k_offsets[:,None] < N_KEYS),other=0.0).to(tl.float32)
        v = tl.load(v_ptrs,mask=(k_offsets[:,None] < N_KEYS),other=0.0)

        # S = q @ k^T * scale -> [Bq, Bk]
        S = tl.dot(q,tl.trans(k)) * scale

        if IS_CAUSAL:
            





