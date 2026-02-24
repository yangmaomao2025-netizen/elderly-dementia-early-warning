#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2数据预处理并生成热力图 (V4 - 45×68)
按要求：
- 高：45行 (3部分×15秒)
- 宽：68列 (4单元×17AU)
- 水平白线：分隔 F2_积极前60秒 | F2_积极后60秒 | F2_中性
- 垂直虚线：分隔4个单元
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
    """数据预处理"""
    print(f"\n处理文件: {csv_file}")
    
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    
    print(f"  原始数据行数: {len(df)}")
    
    df = df[df['success'] == 1].copy()
    print(f"  剔除success=0后: {len(df)}")
    
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    df = df[df['timestamp'] <= target_seconds].copy()
    print(f"  保留{target_seconds}秒后: {len(df)}")
    
    df['second'] = df['timestamp'].astype(int)
    au_cols_clean = [col.strip() for col in AU_R_COLS]
    au_sums = df.groupby('second')[au_cols_clean].sum()
    
    all_seconds = pd.DataFrame({'second': range(target_seconds)})
    au_sums = all_seconds.merge(au_sums.reset_index(), on='second', how='left').fillna(0)
    au_sums = au_sums.drop('second', axis=1)
    
    print(f"  每秒求和后行数: {len(au_sums)}")
    
    return au_sums

def create_unit_transposed(au_sums, start_sec, end_sec):
    """
    创建一个单元的数据（转置版）
    每个单元：15行(秒) × 17列(AU)
    """
    return au_sums.iloc[start_sec:end_sec].values  # 15×17

def main():
    base_dir = "/root/.openclaw/workspace/老年失智人群预警模式科研项目/zcsj"
    output_dir = "/root/.openclaw/workspace/老年失智人群预警模式科研项目/analysis_results/F2_热力图"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("步骤1：数据预处理")
    print("=" * 60)
    
    f2_neutral = preprocess_data(os.path.join(base_dir, "F2_中性.csv"), 60)
    f2_positive = preprocess_data(os.path.join(base_dir, "F2_积极.csv"), 120)
    
    print("\n" + "=" * 60)
    print("步骤2：生成热力图数据 (45×68)")
    print("=" * 60)
    
    # 每个单元：15行(秒) × 17列(AU)
    # 4个单元按列拼接：15行 × 68列
    
    print("\nF2_中性.csv：")
    neutral_units = []
    for i in range(4):
        start_sec = i * 15
        end_sec = (i + 1) * 15
        unit_data = create_unit_transposed(f2_neutral, start_sec, end_sec)  # 15×17
        neutral_units.append(unit_data)
        print(f"  单元{i+1} ({start_sec+1}-{end_sec}s): {unit_data.shape}")
    
    # 按列拼接4个单元：15×68
    neutral_heatmap = np.hstack(neutral_units)
    print(f"  F2_中性热力图: {neutral_heatmap.shape} (15行×68列)")
    
    print("\nF2_积极.csv：")
    
    # 前60秒
    positive_first_units = []
    for i in range(4):
        start_sec = i * 15
        end_sec = (i + 1) * 15
        unit_data = create_unit_transposed(f2_positive.iloc[:60], start_sec, end_sec)
        positive_first_units.append(unit_data)
    positive_first = np.hstack(positive_first_units)  # 15×68
    print(f"  前60秒: {positive_first.shape}")
    
    # 后60秒
    positive_second_units = []
    for i in range(4):
        start_sec = i * 15
        end_sec = (i + 1) * 15
        unit_data = create_unit_transposed(f2_positive.iloc[60:120], start_sec, end_sec)
        positive_second_units.append(unit_data)
    positive_second = np.hstack(positive_second_units)  # 15×68
    print(f"  后60秒: {positive_second.shape}")
    
    # 按行拼接3个部分：45×68
    final_heatmap = np.vstack([positive_first, positive_second, neutral_heatmap])
    print(f"\n最终热力图: {final_heatmap.shape} (45行×68列)")
    print("高=45行, 宽=68列")
    
    # ========== 绘制热力图 ==========
    print("\n" + "=" * 60)
    print("步骤3：绘制热力图")
    print("=" * 60)
    
    # 行标签：3部分 × 15秒 = 45行
    row_labels = []
    # F2_积极前60秒 (1-15秒)
    for sec in range(1, 16):
        row_labels.append(f"P1_{sec}s")
    # F2_积极后60秒 (1-15秒，对应61-75秒)
    for sec in range(1, 16):
        row_labels.append(f"P2_{sec}s")
    # F2_中性 (1-15秒)
    for sec in range(1, 16):
        row_labels.append(f"N_{sec}s")
    
    # 列标签：4单元 × 17AU = 68列
    au_labels = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 
                 'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 
                 'AU20', 'AU23', 'AU25', 'AU26', 'AU45']
    col_labels = []
    for unit in range(1, 5):
        for au in au_labels:
            col_labels.append(f"{au}_U{unit}")
    
    # 绘制
    fig, ax = plt.subplots(figsize=(20, 12))
    
    sns.heatmap(final_heatmap, 
                cmap='RdBu_r',
                center=0,
                xticklabels=col_labels[::2],
                yticklabels=row_labels,
                cbar_kws={'label': 'AU Intensity Sum'},
                ax=ax)
    
    ax.set_title('F2 AU Heatmap (45 × 68)\nHeight: 45 rows (3 parts × 15s) | Width: 68 cols (4 units × 17 AU)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Width: 68 cols (4 units × 17 AU)', fontsize=12)
    ax.set_ylabel('Height: 45 rows (3 parts × 15s)', fontsize=12)
    
    # 水平白线：分隔 F2_积极前60s | F2_积极后60s | F2_中性
    ax.axhline(y=15, color='white', linewidth=2)
    ax.axhline(y=30, color='white', linewidth=2)
    
    # 垂直虚线：分隔4个单元
    for i in range(1, 4):
        ax.axvline(x=i*17, color='white', linewidth=1, linestyle='--')
    
    # 添加标签说明
    ax.text(8.5, 7.5, 'F2_积极\n前60s', ha='center', va='center', fontsize=10, color='black', fontweight='bold')
    ax.text(8.5, 22.5, 'F2_积极\n后60s', ha='center', va='center', fontsize=10, color='black', fontweight='bold')
    ax.text(8.5, 37.5, 'F2_中性', ha='center', va='center', fontsize=10, color='black', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, "F2_heatmap_45x68_v4.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 热力图已保存: {output_path}")
    print(f"  尺寸: 45行 × 68列 (高45, 宽68)")
    
    # 保存数据
    data_path = os.path.join(output_dir, "F2_heatmap_data_45x68_v4.csv")
    pd.DataFrame(final_heatmap, index=row_labels, columns=col_labels).to_csv(data_path)
    print(f"✓ 数据已保存: {data_path}")
    
    print("\n" + "=" * 60)
    print("完成！热力图尺寸: 45行 × 68列 (高45, 宽68)")
    print("水平白线分隔3部分，垂直虚线分隔4单元")
    print("=" * 60)

if __name__ == "__main__":
    main()
