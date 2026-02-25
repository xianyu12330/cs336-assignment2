import math

import torch



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
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):


class FlashAttention2_triton()