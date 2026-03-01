import math
from typing import Any

import torch
import triton
import triton.language as tl



def flash_bwd_recompute_impl(q: torch.Tensor,
                             k: torch.Tensor,
                             v: torch.Tensor,
                             o: torch.Tensor,
                             do: torch.Tensor,
                             L: torch.Tensor,
                             is_causal: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, S, D = q.shape
    K = k.shape[1]
    scale = 1.0 / math.sqrt(D)
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        q_idx = torch.arange(S, device=s.device)[:, None]
        k_idx = torch.arange(K, device=s.device)[None, :]
        causal = q_idx >= k_idx  # []s,k
        s = torch.where(causal[None, :, :], s, torch.full_like(s, -1.0e6))  # [s,k] ->[1,s,k]
    # Pij = exp (Sij − Li)->[b,s,k]
    p = torch.exp(s - L.unsqueeze(-1))
    # dV = P⊤dO->[b,k,d]
    dv = torch.matmul(p.transpose(-1, -2), do)
    # dP = dOV⊤
    dp = torch.matmul(do, v.transpose(-1, -2))
    # dSij = Pij ◦ (dPij − Di)
    Devc = (do * o).sum(-1)  # [b,s]
    ds = p * (dp - Devc.unsqueeze(-1)) * scale
    # dQ = dSK /√d->[b,s,k]
    dq = torch.matmul(ds, k)
    # dK = dS⊤Q/√d,
    dk = torch.matmul(ds.transpose(-1, -2), q)
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class FlashAttention2_PyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
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
        B, S, D = q.shape
        scale = 1.0 / math.sqrt(D)
        O = torch.empty((B, S, D), device=q.device, dtype=q.dtype)
        L = torch.empty((B, S), device=q.device, dtype=q.dtype)
        # ... [这里将是你实现 FlashAttention-2 核心的分块双重循环逻辑] ...
        # 注意：作业明确说明目前可以忽略 is_causal=True 的掩码逻辑
        # 从0~seq_len每次增长一块大小b_r
        for i in range(0, S, B_r):
            # 取所有行的i:i+B_r列
            q_block = q[:, i:i + B_r, :]
            # 累计输出
            O_acc = torch.zeros((B, B_r, D), device=q.device, dtype=torch.float32)
            # 累计的 softmax 分母-因为每次都会更新最大值 m之前的 exp 需要重新缩放。
            l = torch.zeros((B, B_r), device=q.device, dtype=torch.float32)
            # 当前块累计的最大值,每个 query 位置目前为止看到的 score 最大值
            m = torch.full((B, B_r), -float('inf'), device=q.device, dtype=torch.float32)
            for j in range(0, S, B_c):
                # 加载分块的kv到速度快的
                k_j = k[:, j:j + B_c, :]
                v_j = v[:, j:j + B_c, :]
                ## S = q @ k^T * scale -> (B, B_r, B_c)
                pre_att_score = torch.matmul(q_block, k_j.transpose(-1, -2)) * scale
                # softmax update  ->(沿着最后一个维度压缩->(B,B_r)
                m_new = torch.maximum(m, pre_att_score.max(dim=-1).values)
                # exp(S - m_new) ->(B,B_r,B_c)
                p_j = torch.exp(pre_att_score - m_new.unsqueeze(-1))
                # l_new = exp(m - m_new) * l + rowsum(P_tilde)
                # (B,B_r)
                l_new = torch.exp(m - m_new) * l + p_j.sum(dim=-1)
                O_acc = (torch.exp(m - m_new)).unsqueeze(-1) * O_acc + torch.matmul(p_j, v_j)

                m, l = m_new, l_new
            o_i = O_acc / l.unsqueeze(-1)
            O[:, i:i + B_r, :] = o_i.to(q.device)
            # L = logsumexp(S_row) = m + log(l)
            L[:, i:i + B_r] = m + torch.log(l)
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
    def backward(ctx: Any, do: Any):
        (L, q, k, v, o) = ctx.saved_tensors
        dq, dk, dv = flash_bwd_recompute_impl(q, k, v, o, do, L, ctx.is_causal)
        return dq, dk, dv, None


@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,  # 指向数据在显存中起始位置的基指针
        O_ptr, L_ptr,  # Output输出矩阵和logsumexp辅助矩阵的指针
        stride_qb,  # 移动到下一个 Batch 需要跳过的距离。
        stride_qq,  # 移动到下一个 Query 序列位置需要跳过的距离。
        stride_qd,  # 移动到下一个特征维度需要跳过的距离。
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,  # Output的 strides
        stride_lb, stride_lq,  # Logsumexp的 strides
        N_QUERIES: tl.constexpr,  # 相当于总大小
        N_KEYS: tl.constexpr,  # query,key/value序列的长度
        scale,
        D: tl.constexpr,  # 每个头的维度
        Q_TILE_SIZE: tl.constexpr,  # Bq，分块长度，每个 program 处理的数据块大小
        K_TILE_SIZE: tl.constexpr,  # Bk
        IS_CAUSAL: tl.constexpr
):
    # 当前这段代码负责处理的块的坐标。
    pid_q = tl.program_id(0)
    # 表示当前是第几个 Batch
    pid_b = tl.program_id(1)
    # 每个 program instance 负责：一个 batch 的一个 Q tile
    # 每个 block 计算：Q[pid_b, pid_q_tile, :]，每个块的内存范围
    q_offsets = pid_q * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    d_offsets = tl.arange(0, D)
    # 构造地址Q[pid_b, q_offsets[i], d_offsets[j]]，最终是【bq，d】的指针矩阵
    q_ptrs = Q_ptr \
             + pid_b * stride_qb \
             + q_offsets[:, None] * stride_qq \
             + d_offsets[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(q_offsets[:, None] < N_QUERIES), other=0.0).to(tl.float32)
    # 初始化 Online Softmax 状态
    ## 当前最大值
    m = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)  # [bq]
    ## 当前 softmax 分母
    l = tl.zeros((Q_TILE_SIZE,), tl.float32)
    ## 当前加权和
    acc = tl.zeros((Q_TILE_SIZE, D), tl.float32)
    # 每次处理一个 K block
    for kb in range(0, N_KEYS, K_TILE_SIZE):
        k_offsets = kb + tl.arange(0, K_TILE_SIZE)
        k_ptrs = K_ptr + pid_b * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd
        v_ptrs = V_ptr + pid_b * stride_vb + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=(k_offsets[:, None] < N_KEYS), other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=(k_offsets[:, None] < N_KEYS), other=0.0)

        # S = q @ k^T * scale -> [Bq, Bk]
        S = tl.dot(q, tl.trans(k)) * scale
        # 强制 attention 只看过去 token
        if IS_CAUSAL:
            # q，k的相对位置
            q_ads = q_offsets[:, None]  # [bq,1]
            k_ads = k_offsets[None, :]  # [1,bk]
            causal = q_ads >= k_ads
            S = tl.where(causal, S, -1.0e6)

        # online softmax update
        m_new = tl.maximum(m, tl.max(S, axis=1))
        # 计算当前块的 exp
        p = tl.exp(S - m_new[:, None])
        # 修正旧分母
        alpha = tl.exp(m - m_new)
        l_new = alpha * l + tl.sum(p, axis=1)
        # 更新输出累加
        p = p.to(v.dtype)
        acc = alpha[:, None] * acc + tl.dot(p, v)
        # 计算最终输出
        m = m_new
        l = l_new
    o = acc / l[:, None]
    o = o.to(tl.float32)
    o_ptrs = O_ptr + pid_b * stride_ob + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od
    tl.store(o_ptrs, o, mask=(q_offsets[:, None] < N_QUERIES))

    L_out = m + tl.log(l)  # [Bq]
    l_ptrs = L_ptr + pid_b * stride_lb + q_offsets * stride_lq
    tl.store(l_ptrs, L_out, mask=(q_offsets < N_QUERIES))


@triton.jit
def flash_bwd_kernel(
        # --- 前向输入与输出指针 ---
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,  # L_ptr 是前向保存的 logsumexp

        # --- 反向梯度指针 ---
        dO_ptr,  # 上游传来的输出梯度
        dQ_ptr, dK_ptr, dV_ptr,  # 我们需要计算的目标梯度
        Delta_ptr,  # 预计算的常数向量 Delta = sum(dO * O, axis=-1)

        # --- 各种 Stride (这里假设 Q, O, dO, dQ 形状相同，共享 stride 以简化参数) ---
        stride_qb, stride_qq, stride_qd, #*b-》batch跨度，*q-》序列跨度，*d-》特征跨度
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,  # Output的 strides
        stride_lb, stride_lq,  # Logsumexp的 strides

        # --- 维度与 Meta 参数 ---
        N_QUERIES: tl.constexpr,  # 相当于总大小
        N_KEYS: tl.constexpr,  # query,key/value序列的长度
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,  # Bq，分块长度，每个 program 处理的数据块大小
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr
):
    """
    FlashAttention-2 Backward Pass Kernel
    """
    # =================================================================
    # TODO 1: 获取 Program ID 与偏移量 (Offsets)
    # =================================================================
    # 1. 获取当前负责的 Query 块 ID (pid_q) 和 Batch ID (pid_b)
    # 2. 生成当前 Q 块的行偏移量 q_offsets [Bq]
    # 3. 生成特征维度的列偏移量 d_offsets [D]
    pid_q = tl.program_id(0)
    pid_b = tl.program_id(1)
    q_offset = pid_q * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE) #[bq,]
    b_offsets = tl.arange(0, D)#[D,]

    # =================================================================
    # TODO 2: 加载当前 Q 块相关的变量到 SRAM
    # =================================================================
    # 1. 计算 q_ptrs, do_ptrs, o_ptrs，并加载 q, do, o (别忘了 mask 越界保护)
    # 2. 计算 l_ptrs (注意 L 是 1D 数组 [Bq]) 并加载 L_i
    # 3. 计算 delta_ptrs 并加载预计算的 Delta_i [Bq]
    # 4. 在 SRAM 中初始化 dQ 累加器: dq = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    q_ptrs = Q_ptr + pid_b * stride_qb \
             + q_offset[:, None] * stride_qq \
             + b_offsets[None, :] * stride_qd #[bq,d]
    # 上游传来的输出梯度
    do_ptrs = dO_ptr + pid_b * stride_ob \
              + q_offset[:, None] * stride_oq \
              + b_offsets[None, :] * stride_od
    o_ptrs = O_ptr + pid_b * stride_ob \
             + q_offset[:, None] * stride_oq \
             + b_offsets[None, :] * stride_od
    # 把数据拉进 SRAM
    q = tl.load(q_ptrs, mask=(q_offset[:, None] < N_QUERIES), other=0.0)
    do = tl.load(do_ptrs, mask=(q_offset[:, None] < N_QUERIES), other=0.0)
    o = tl.load(o_ptrs, mask=(q_offset[:, None] < N_QUERIES), other=0.0)
    # 2. 计算一维向量的指针 (L, Delta) -> 注意这里没有 d_offsets！
    l_ptrs = L_ptr + pid_b * stride_lb + q_offset * stride_lq
    delta_ptrs = Delta_ptr + pid_b * stride_lb + q_offset * stride_lq
    # 3.加载 L_i 和 Delta_i
    #Li
    L_i = tl.load(l_ptrs, mask=(q_offset < N_QUERIES), other=0.0)
    Delta_i = tl.load(delta_ptrs, mask=(q_offset < N_QUERIES), other=0.0)
    # 3. 初始化 dQ 累加器 (全 0)
    dq = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    # 1. 计算 l_ptrs 并加载 L_i [Bq]
    # =================================================================
    # 内层循环：遍历所有的 K, V 块
    # =================================================================
    for kb in tl.range(0, N_KEYS, K_TILE_SIZE):
        # TODO 3: 计算当前 K/V 块的偏移量 k_offsets [Bk]
        k_offsets = kb + tl.arange(0, K_TILE_SIZE)

        # TODO 4: 加载当前 K/V 块到 SRAM
        # 1. 计算 k_ptrs, v_ptrs
        # 2. 加载 k, v (使用 mask 保护)
        k_ptrs = K_ptr + pid_b * stride_kb + k_offsets[:, None] * stride_kk + b_offsets[None, :] * stride_kd
        v_ptrs = V_ptr + pid_b * stride_vb + k_offsets[:, None] * stride_vk + b_offsets[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=(k_offsets[:, None] < N_KEYS), other=0.0)
        v = tl.load(v_ptrs, mask=(k_offsets[:, None] < N_KEYS), other=0.0)
        # =================================================================
        # TODO 5: 重算局部 Attention 分数 P
        # =================================================================
        # 1. 计算 S = (q @ k^T) * scale
        # 2. 如果 IS_CAUSAL 为 True，生成 causal_mask 并将越界处的 S 设为 -1.0e6
        # 3. 利用保存的 L_i 重算 P: P = tl.exp(S - L_i[:, None])  # [Bq, Bk]
        S = tl.dot(q, tl.trans(k)) * scale
        if IS_CAUSAL:
            causal_mask = q_offset[:, None] >= k_offsets[None, :]
            S = tl.where(causal_mask, S, -1.0e6)
        P = tl.exp(S - L_i[:, None])
        # =================================================================
        # TODO 6: 计算 dV 并原子累加到 HBM
        # =================================================================
        # 1. 数学公式: dV_local = P^T @ dO
        #    提示: 需要转置 P，即 tl.dot(tl.trans(P), do)
        # 2. 计算 dv_ptrs (和 v_ptrs 结构一样)
        # 3. 使用 tl.atomic_add(dv_ptrs, dv_local, mask=...) 将结果原子累加到全局显存
        # 为了让 Tensor Core 高效计算，通常把 P 强转为和 do 相同的精度 (如 fp16)
        P_type = P.to(do.dtype)
        dv_local = tl.dot(tl.trans(P_type), do)  # [Bk, D]

        dv_ptrs = dV_ptr + pid_b * stride_vb + k_offsets[:, None] * stride_vk + b_offsets[None, :] * stride_vd
        tl.atomic_add(dv_ptrs, dv_local, mask=(k_offsets[:, None] < N_KEYS))
        # =================================================================
        # TODO 7: 计算 dP 和 dS
        # =================================================================
        # 1. dP = do @ v^T  # [Bq, Bk]
        # 2. dS = P * (dP - Delta_i[:, None]) * scale  # 逐元素相乘，注意广播 Delta_i
        #    提示: 为了精度，建议先转成 float32 再做乘法
        dp = tl.dot(do, tl.trans(v))
        ds = P * (dp - Delta_i[:, None]) * scale

        # =================================================================
        # TODO 8: 在 SRAM 中累加 dQ
        # =================================================================
        # 1. 数学公式: dQ += dS @ k
        # 2. 注意：这里直接原地累加到 TODO 2 中初始化的 dq 变量上即可，不需要 atomic_add
        ds_type = ds.to(k.dtype)
        dq += tl.dot(ds_type, k)

        # =================================================================
        # TODO 9: 计算 dK 并原子累加到 HBM
        # =================================================================
        # 1. 数学公式: dK_local = dS^T @ q
        #    提示: 需要转置 dS
        # 2. 计算 dk_ptrs (和 k_ptrs 结构一样)
        # 3. 使用 tl.atomic_add(dk_ptrs, dk_local, mask=...) 将结果原子累加到全局显存
        dk_local = tl.dot(tl.trans(ds_type), q)
        # 构建 dK 的目标指针 (形状与 K 相同)
        dk_ptrs = dK_ptr + pid_b * stride_kb + k_offsets[:, None] * stride_kk + b_offsets[None, :] * stride_kd
        tl.atomic_add(dk_ptrs, dk_local, mask=(k_offsets[:, None] < N_KEYS))
    # =================================================================
    # TODO 10: 循环结束，将完全算好的 dQ 存回 HBM
    # =================================================================
    # 1. 计算 dq_ptrs (和 q_ptrs 结构一样)
    # 2. 使用 tl.store 将 dq 写入全局显存 (注意 mask 越界保护)
    dq_ptrs = dQ_ptr + pid_b * stride_qb + q_offset[:, None] * stride_qq + b_offsets[None, :] * stride_qd
    tl.store(dq_ptrs, dq, mask=(q_offset[:, None] < N_QUERIES))


class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        B, S, D = q.shape
        scale = 1.0 / math.sqrt(D)
        Bq = 32
        Bk = 32
        o = torch.empty((B, S, D), device=q.device, dtype=q.dtype)
        L = torch.empty((B, S), device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(S, Bq), B)
        flash_fwd_kernel[grid](
            q, k, v,
            o, L,
            q.stride()[0], q.stride()[1], q.stride()[2],
            k.stride()[0], k.stride()[1], k.stride()[2],
            v.stride()[0], v.stride()[1], v.stride()[2],
            o.stride()[0], o.stride()[1], o.stride()[2],
            L.stride()[0], L.stride()[1],
            N_QUERIES=S,
            N_KEYS=k.shape[1],
            scale=scale,
            D=D, Q_TILE_SIZE=Bq,
            K_TILE_SIZE=Bk,
            IS_CAUSAL=is_causal,
            num_warps=4
        )
        # save for backward
        ctx.save_for_backward(L, q, k, v, o)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, grad_output):
        """
        接收上游传回来的梯度 dO (grad_output)
        返回关于输入 Q, K, V 的梯度 dQ, dK, dV
        """
        # 1. 提取前向传播保存的张量
        L, Q, K, V, O = ctx.saved_tensors

        # 将 grad_output 变得连续（消除内存碎片，防止底层步长错乱）
        # 并确保它是能进行底层运算的半精度或单精度浮点数
        dO = grad_output.contiguous()

        # 2. 获取维度信息 (假设输入形状为 [Batch, Seq_Len, Dim])
        # 如果你的输入包含 Num_Heads，你需要先把它 reshape 成 [Batch * Num_Heads, Seq_Len, Dim]
        Batch, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape

        scale = 1.0 / math.sqrt(D)

        # 3. 在 PyTorch 层预计算 Delta
        # Delta = sum(dO * O, axis=-1)，形状为 [Batch, Seq_Len]
        # 注意保持精度为 float32 以防止数值溢出
        Delta = torch.sum(dO.float() * O.float(), dim=-1).contiguous()

        # 4. 初始化梯度张量
        # dQ 可以不初始化，因为我们在内核里是覆盖写 (store)
        dQ = torch.empty_like(Q)
        # dK 和 dV 必须初始化为 0，因为我们在内核里用的是原子加 (atomic_add)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        # 5. 定义 Triton 的 Block 尺寸与 Grid
        # 这里的 TILE_SIZE 必须与你前向/反向 Kernel 里的 tl.constexpr 对应
        Q_TILE_SIZE = 32
        K_TILE_SIZE = 32

        # 网格划分：有多少个 Q 块，就有多少个一维的并发任务；
        # 第二个维度按 Batch 进行并行计算。
        grid = lambda meta: (
            triton.cdiv(N_QUERIES, meta['Q_TILE_SIZE']),
            Batch
        )

        # 6. 调用 Triton 反向传播 Kernel
        flash_bwd_kernel[grid](
            # 前向输入与输出
            Q, K, V,
            O, L,
            # 梯度与预计算变量
            dO,
            dQ, dK, dV,
            Delta,
            # Strides (这里直接调用 PyTorch 张量的 .stride() 方法)
            Q.stride()[0], Q.stride()[1], Q.stride()[2],
            K.stride()[0], K.stride()[1], K.stride()[2],
            V.stride()[0], V.stride()[1], V.stride()[2],
            O.stride()[0], O.stride()[1], O.stride()[2],
            L.stride()[0], L.stride()[1],  # L 是二维的 [Batch, Seq_Len]
            # 维度与 Meta 参数
            N_QUERIES=N_QUERIES,
            N_KEYS=N_KEYS,
            scale=scale,
            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=ctx.is_causal,
        )

        # 7. 返回梯度
        # 返回值的数量和顺序必须与 forward 函数的输入参数完全一致！
        # forward 签名: (ctx, Q, K, V, is_causal)
        # 所以返回 dQ, dK, dV，并且给没有梯度的参数 is_causal 返回 None
        return dQ, dK, dV, None





