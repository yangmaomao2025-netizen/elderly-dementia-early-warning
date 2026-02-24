#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2数据预处理并生成热力图 (V3 - 68×45)
按要求：
- F2_中性：4个单元(17×15)按列方向拼接 → 68行×15列
- F2_积极：两个60秒各68×15，按行方向拼接 → 68行×30列  
- 最终：按行方向拼接 → 68行×45列
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# AU强度列名
AU_R_COLS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
             'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
             'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

def preprocess_data(csv_file, target_seconds):
    """
    数据预处理：
    1. 剔除success为0的行
    2. 保留目标秒数的数据（从首尾剔除超出部分）
    3. 对连续1秒的AU强度值求和
    """
    print(f"\n处理文件: {csv_file}")
    
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    
    print(f"  原始数据行数: {len(df)}")
    
    # 剔除success为0的行
    df = df[df['success'] == 1].copy()
    print(f"  剔除success=0后: {len(df)}")
    
    # 确保timestamp从0开始
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    
    # 保留目标秒数的数据
    df = df[df['timestamp'] <= target_seconds].copy()
    
    # 从尾部剔除超出部分（如果数据超过目标秒数）
    # 实际上已经通过<=筛选了，但如果数据点不均匀，可能需要额外处理
    
    print(f"  保留{target_seconds}秒后: {len(df)}")
    
    # 添加秒数列（向下取整）
    df['second'] = df['timestamp'].astype(int)
    
    # 清理AU列名
    au_cols_clean = [col.strip() for col in AU_R_COLS]
    
    # 对连续1秒的AU强度值求和
    au_sums = df.groupby('second')[au_cols_clean].sum()
    
    # 确保有目标秒数行
    all_seconds = pd.DataFrame({'second': range(target_seconds)})
    au_sums = all_seconds.merge(au_sums.reset_index(), on='second', how='left').fillna(0)
    au_sums = au_sums.drop('second', axis=1)
    
    print(f"  每秒求和后行数: {len(au_sums)}")
    
    return au_sums

def create_unit(au_sums, start_sec, end_sec):
    """
    创建一个单元的数据
    每个单元：17行(AU) × 15列(秒)
    """
    return au_sums.iloc[start_sec:end_sec].values.T  # 转置：行=AU，列=时间

def main():
    base_dir = "/root/.openclaw/workspace/老年失智人群预警模式科研项目/zcsj"
    output_dir = "/root/.openclaw/workspace/老年失智人群预警模式科研项目/analysis_results/F2_热力图"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("步骤1：数据预处理")
    print("=" * 60)
    
    # 数据预处理
    f2_neutral = preprocess_data(os.path.join(base_dir, "F2_中性.csv"), 60)
    f2_positive = preprocess_data(os.path.join(base_dir, "F2_积极.csv"), 120)
    
    print("\n" + "=" * 60)
    print("步骤2：生成热力图数据 (68×45)")
    print("=" * 60)
    
    # ========== F2_中性：4个单元按列方向拼接 → 68行×15列 ==========
    print("\nF2_中性.csv：")
    neutral_units = []
    for i in range(4):
        start_sec = i * 15
        end_sec = (i + 1) * 15
        unit_data = create_unit(f2_neutral, start_sec, end_sec)  # 17×15
        neutral_units.append(unit_data)
        print(f"  单元{i+1} ({start_sec+1}-{end_sec}s): {unit_data.shape}")
    
    # 按列方向拼接（垂直堆叠）：(17×4) × 15 = 68×15
    neutral_heatmap = np.vstack(neutral_units)
    print(f"  F2_中性热力图: {neutral_heatmap.shape} (68行×15列)")
    
    # ========== F2_积极：两个60秒，各生成68×15，按行方向拼接 → 68×30 ==========
    print("\nF2_积极.csv：")
    
    # 前60秒
    positive_first_units = []
    for i in range(4):
        start_sec = i * 15
        end_sec = (i + 1) * 15
        unit_data = create_unit(f2_positive.iloc[:60], start_sec, end_sec)
        positive_first_units.append(unit_data)
    positive_first = np.vstack(positive_first_units)  # 68×15
    print(f"  前60秒: {positive_first.shape}")
    
    # 后60秒
    positive_second_units = []
    for i in range(4):
        start_sec = i * 15
        end_sec = (i + 1) * 15
        unit_data = create_unit(f2_positive.iloc[60:120], start_sec, end_sec)
        positive_second_units.append(unit_data)
    positive_second = np.vstack(positive_second_units)  # 68×15
    print(f"  后60秒: {positive_second.shape}")
    
    # 按行方向拼接（水平拼接）：68 × (15+15) = 68×30
    positive_heatmap = np.hstack([positive_first, positive_second])
    print(f"  F2_积极热力图: {positive_heatmap.shape} (68行×30列)")
    
    # ========== 最终：按行方向拼接 → 68×45 ==========
    final_heatmap = np.hstack([positive_heatmap, neutral_heatmap])
    print(f"\n最终热力图: {final_heatmap.shape} (68行×45列)")
    print("宽=45列, 高=68行")
    
    # ========== 绘制热力图 ==========
    print("\n" + "=" * 60)
    print("步骤3：绘制热力图")
    print("=" * 60)
    
    # 行标签：4个单元 × 17个AU = 68行
    au_labels = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 
                 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 
                 'AU20', 'AU23', 'AU25', 'AU26', 'AU45']
    row_labels = []
    for unit in range(1, 5):
        for au in au_labels:
            row_labels.append(f"{au}_U{unit}")
    
    # 列标签 - 修正：只有45列
    col_labels = []
    # 总共45列：前30列(F2_积极) + 后15列(F2_中性)
    # 每部分包含4个单元，每单元包含15秒
    for i in range(45):
        col_labels.append(f"Col_{i+1}")
    
    # 绘制
    fig, ax = plt.subplots(figsize=(24, 16))
    
    sns.heatmap(final_heatmap, 
                cmap='RdBu_r',
                center=0,
                xticklabels=col_labels[::3],  # 每隔3个显示
                yticklabels=row_labels,
                cbar_kws={'label': 'AU Intensity Sum'},
                ax=ax)
    
    ax.set_title('F2 AU Heatmap (68 × 45)\nRows: 4 Units × 17 AU = 68 | Cols: 3 Parts × 4 Units × 15s = 45', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (3 parts: P1+P2+N) × 4 units × 15s = 45 cols', fontsize=12)
    ax.set_ylabel('AU Units (4 units × 17 AU = 68 rows)', fontsize=12)
    
    # 垂直分隔线：分隔 F2_积极前60s | F2_积极后60s | F2_中性
    ax.axvline(x=60, color='white', linewidth=2)
    ax.axvline(x=30, color='white', linewidth=1, linestyle='--')
    ax.axvline(x=90, color='white', linewidth=1, linestyle='--')
    
    # 水平分隔线：分隔4个单元
    for i in range(1, 4):
        ax.axhline(y=i*17, color='white', linewidth=1, linestyle='--')
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, "F2_heatmap_68x45_v3.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 热力图已保存: {output_path}")
    print(f"  尺寸: 68行 × 45列 (高68, 宽45)")
    
    # 保存数据
    data_path = os.path.join(output_dir, "F2_heatmap_data_68x45.csv")
    pd.DataFrame(final_heatmap, index=row_labels, columns=col_labels).to_csv(data_path)
    print(f"✓ 数据已保存: {data_path}")
    
    print("\n" + "=" * 60)
    print("完成！热力图尺寸: 68行 × 45列 (高68, 宽45)")
    print("=" * 60)

if __name__ == "__main__":
    main()
