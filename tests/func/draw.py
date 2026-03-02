import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_benchmark_results(csv_file="benchmark_results.csv", output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(csv_file):
        print(f"找不到文件 {csv_file}，请确保路径正确。")
        return
    
    # 读取数据
    df = pd.read_csv(csv_file)

    # 1. 清理列名前后的空格
    df.columns = df.columns.str.strip()

    time_cols = [
        'pytorch_fwd(ms)', 'triton_fwd(ms)', 
        'pytorch_bwd(ms)', 'triton_bwd(ms)', 
        'pytorch_e2e(ms)', 'triton_e2e(ms)'
    ]
    
    missing_cols = [col for col in time_cols if col not in df.columns]
    if missing_cols:
        print(f"警告：CSV 中找不到以下列：{missing_cols}")
        print(f"当前 CSV 列名是：{df.columns.tolist()}")
        return

    # 🌟 2. 核心修复：极致的数据清洗 🌟
    for col in time_cols:
        # 先全部当做字符串处理
        s = df[col].astype(str)
        # 剔除 'ms' 字符和空白
        s = s.str.replace('ms', '', regex=False).str.strip()
        # 将 OOM 替换为空值
        s = s.replace(['OOM', 'Error/OOM', 'nan', 'None'], np.nan)
        # 转换为浮点数
        df[col] = pd.to_numeric(s, errors='coerce')

    # 计算加速比
    df['Speedup_Fwd'] = df['pytorch_fwd(ms)'] / df['triton_fwd(ms)']
    df['Speedup_Bwd'] = df['pytorch_bwd(ms)'] / df['triton_bwd(ms)']
    df['Speedup_E2E'] = df['pytorch_e2e(ms)'] / df['triton_e2e(ms)']

    dtypes = df['Dtype'].unique()
    dims = df['Dim'].unique()

    print(f"数据加载并清洗完成！开始生成图表，保存至 '{output_dir}/' 目录下...")

    for dtype in dtypes:
        for dim in dims:
            subset = df[(df['Dtype'] == dtype) & (df['Dim'] == dim)].sort_values('Seq_len')
            
            # 🌟 3. 保护逻辑：如果这个组合的数据全为空，直接跳过 🌟
            if subset[time_cols].dropna(how='all').empty:
                print(f"跳过 Dtype={dtype}, Dim={dim}，因为该组全是 OOM 或无有效数据。")
                continue

            seq_lens = subset['Seq_len']
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle(f'FlashAttention-2 vs PyTorch\nDtype={dtype}, Head Dim={dim}', fontsize=16, fontweight='bold')

            metrics = [
                ('Forward', 'pytorch_fwd(ms)', 'triton_fwd(ms)', 'Speedup_Fwd'),
                ('Backward', 'pytorch_bwd(ms)', 'triton_bwd(ms)', 'Speedup_Bwd'),
                ('End-to-End', 'pytorch_e2e(ms)', 'triton_e2e(ms)', 'Speedup_E2E')
            ]

            for i, (name, pt_col, fa2_col, speedup_col) in enumerate(metrics):
                # --- 上排：延迟对比图 ---
                ax_lat = axes[0, i]
                # 过滤掉 NaN 数据点画线，防止断图报错
                valid_pt = subset.dropna(subset=[pt_col])
                valid_fa2 = subset.dropna(subset=[fa2_col])
                
                if not valid_pt.empty:
                    ax_lat.plot(valid_pt['Seq_len'], valid_pt[pt_col], marker='o', linestyle='-', color='red', label='PyTorch')
                if not valid_fa2.empty:
                    ax_lat.plot(valid_fa2['Seq_len'], valid_fa2[fa2_col], marker='*', linestyle='-', color='green', label='Triton FA2')
                
                ax_lat.set_title(f'{name} Latency')
                ax_lat.set_xlabel('Sequence Length')
                ax_lat.set_ylabel('Latency (ms)')
                ax_lat.set_xscale('log', base=2)
                
                # 只有当包含有效数据时，才设置 Y 轴为 log，防止报错
                if (not valid_pt.empty) or (not valid_fa2.empty):
                    ax_lat.set_yscale('log') 
                    
                ax_lat.grid(True, which="both", ls="--", alpha=0.5)
                ax_lat.legend()

                # --- 下排：加速比图 ---
                ax_speedup = axes[1, i]
                valid_speedup = subset.dropna(subset=[speedup_col])
                
                if not valid_speedup.empty:
                    ax_speedup.plot(valid_speedup['Seq_len'], valid_speedup[speedup_col], marker='s', linestyle='-', color='blue')
                
                ax_speedup.set_title(f'{name} Speedup')
                ax_speedup.set_xlabel('Sequence Length')
                ax_speedup.set_ylabel('Speedup (x)')
                ax_speedup.set_xscale('log', base=2)
                ax_speedup.axhline(1.0, color='black', linestyle='--', alpha=0.7, label='1x Baseline')
                ax_speedup.grid(True, which="both", ls="--", alpha=0.5)
                ax_speedup.legend()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # 保存图表
            filename = f"benchmark_{str(dtype).replace('torch.', '')}_dim{dim}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300)
            plt.close()
            
            print(f"✅ 成功生成图表: {filepath}")

if __name__ == "__main__":
    plot_benchmark_results(csv_file="benchmark_results.csv")