#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F2数据预处理并生成热力图 (V7 - 75×68)
按要求：
- 高：75行 (5部分×15秒)
- 宽：68列 (4单元×17AU)
- 水平白线：分隔 F2_中性 | F2_积极前60秒 | F2_积极后60秒 | F2_悲伤前60秒 | F2_悲伤后60秒
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

def preprocess_data(csv_file, start_sec, end_sec):
    """
    数据预处理：
    1. 剔除success为0的行
    2. 保留指定秒数范围的数据
    3. 对连续1秒的AU强度值求和
    """
    print(f"\n处理文件: {csv_file}")
    print(f"  保留范围: {start_sec}-{end_sec}秒")
    
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    
    print(f"  原始数据行数: {len(df)}")
    
    # 剔除success为0的行
    df = df[df['success'] == 1].copy()
    print(f"  剔除success=0后: {len(df)}")
    
    # 确保timestamp从0开始
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    
    # 保留指定秒数范围的数据
    df = df[(df['timestamp'] >= start_sec) & (df['timestamp'] < end_sec)].copy()
    print(f"  保留{start_sec}-{end_sec}秒后: {len(df)}")
    
    # 重新设置timestamp从0开始
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    
    # 添加秒数列（向下取整）
    df['second'] = df['timestamp'].astype(int)
    
    # 清理AU列名
    au_cols_clean = [col.strip() for col in AU_R_COLS]
    
    # 对连续1秒的AU强度值求和
    au_sums = df.groupby('second')[au_cols_clean].sum()
    
    # 确保有目标秒数行
    target_seconds = end_sec - start_sec
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
    
    # F2_中性：0-60秒
    f2_neutral = preprocess_data(os.path.join(base_dir, "F2_中性.csv"), 0, 60)
    
    # F2_积极：0-120秒
    f2_positive = preprocess_data(os.path.join(base_dir, "F2_积极.csv"), 0, 120)
    
    # F2_悲伤：61-180秒
    f2_sad = preprocess_data(os.path.join(base_dir, "F2_悲伤.csv"), 61, 181)
    
    print("\n" + "=" * 60)
    print("步骤2：生成热力图数据 (75×68)")
    print("=" * 60)
    
    # 每个单元：15行(秒) × 17列(AU)
    # 4个单元按列拼接：15行 × 68列
    
    # ========== F2_中性：15×68 ==========
    print("\nF2_中性.csv：")
    neutral_units = []
    for i in range(4):
        start_sec = i * 15
        end_sec = (i + 1) * 15
        unit_data = create_unit_transposed(f2_neutral, start_sec, end_sec)  # 15×17
        neutral_units.append(unit_data)
        print(f"  单元{i+1} ({start_sec+1}-{end_sec}s): {unit_data.shape}")
    neutral_heatmap = np.hstack(neutral_units)  # 15×68
    print(f"  F2_中性热力图: {neutral_heatmap.shape}")
    
    # ========== F2_积极前60秒：15×68 ==========
    print("\nF2_积极.csv 前60秒：")
    positive_first_units = []
    for i in range(4):
        start_sec = i * 15
        end_sec = (i + 1) * 15
        unit_data = create_unit_transposed(f2_positive.iloc[:60], start_sec, end_sec)
        positive_first_units.append(unit_data)
    positive_first = np.hstack(positive_first_units)  # 15×68
    print(f"  F2_积极前60秒: {positive_first.shape}")
    
    # ========== F2_积极后60秒：15×68 ==========
    print("\nF2_积极.csv 后60秒：")
    positive_second_units = []
    for i in range(4):
        start_sec = i * 15
        end_sec = (i + 1) * 15
        unit_data = create_unit_transposed(f2_positive.iloc[60:120], start_sec, end_sec)
        positive_second_units.append(unit_data)
    positive_second = np.hstack(positive_second_units)  # 15×68
    print(f"  F2_积极后60秒: {positive_second.shape}")
    
    # ========== F2_悲伤前60秒：15×68 ==========
    print("\nF2_悲伤.csv 前60秒：")
    sad_first_units = []
    for i in range(4):
        start_sec = i * 15
        end_sec = (i + 1) * 15
        unit_data = create_unit_transposed(f2_sad.iloc[:60], start_sec, end_sec)
        sad_first_units.append(unit_data)
    sad_first = np.hstack(sad_first_units)  # 15×68
    print(f"  F2_悲伤前60秒: {sad_first.shape}")
    
    # ========== F2_悲伤后60秒：15×68 ==========
    print("\nF2_悲伤.csv 后60秒：")
    sad_second_units = []
    for i in range(4):
        start_sec = i * 15
        end_sec = (i + 1) * 15
        unit_data = create_unit_transposed(f2_sad.iloc[60:120], start_sec, end_sec)
        sad_second_units.append(unit_data)
    sad_second = np.hstack(sad_second_units)  # 15×68
    print(f"  F2_悲伤后60秒: {sad_second.shape}")
    
    # ========== 按行拼接5个部分：75×68 ==========
    # 顺序：F2_中性 | F2_积极前60秒 | F2_积极后60秒 | F2_悲伤前60秒 | F2_悲伤后60秒
    final_heatmap = np.vstack([neutral_heatmap, positive_first, positive_second, sad_first, sad_second])
    print(f"\n最终热力图: {final_heatmap.shape} (75行×68列)")
    print("高=75行, 宽=68列")
    
    # ========== 绘制热力图 ==========
    print("\n" + "=" * 60)
    print("步骤3：绘制热力图")
    print("=" * 60)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(20, 16))
    
    sns.heatmap(final_heatmap, 
                cmap='RdBu_r',
                center=0,
                xticklabels=False,
                yticklabels=False,
                cbar_kws={'label': 'AU Intensity Sum'},
                ax=ax)
    
    # 设置x轴刻度：在每个单元的中心显示标签
    unit_centers = [8.5, 25.5, 42.5, 59.5]
    unit_labels = ['Unit 1\n(1-15s)', 'Unit 2\n(16-30s)', 'Unit 3\n(31-45s)', 'Unit 4\n(46-60s)']
    ax.set_xticks(unit_centers)
    ax.set_xticklabels(unit_labels, fontsize=10)
    
    # 设置y轴刻度：在每个部分的中心显示标签
    # 5个部分，每部分15行，中心位置：7.5, 22.5, 37.5, 52.5, 67.5
    part_centers = [7.5, 22.5, 37.5, 52.5, 67.5]
    part_labels = ['N\n(Neutral)', 'P1\n(Positive 0-60s)', 'P2\n(Positive 60-120s)', 
                   'S1\n(Sad 61-120s)', 'S2\n(Sad 121-180s)']
    ax.set_yticks(part_centers)
    ax.set_yticklabels(part_labels, fontsize=10, rotation=0)
    
    ax.set_title('F2 AU Heatmap (75 × 68)\nHeight: 75 rows (5 parts × 15s) | Width: 68 cols (4 units × 17 AU)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Width: 68 cols (4 units × 17 AU)', fontsize=10)
    ax.set_ylabel('Height: 75 rows (N | P1 | P2 | S1 | S2)', fontsize=10)
    
    # 水平白线：分隔5个部分（在行边界）
    # 行0-14是N，行15-29是P1，行30-44是P2，行45-59是S1，行60-74是S2
    # 分隔线在行的边界：14.5, 29.5, 44.5, 59.5
    for y in [14.5, 29.5, 44.5, 59.5]:
        ax.axhline(y=y, color='white', linewidth=2)
    
    # 垂直虚线：分隔4个单元（在列边界）
    for x in [16.5, 33.5, 50.5]:
        ax.axvline(x=x, color='white', linewidth=1, linestyle='--')
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, "F2_heatmap_75x68_v7.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ 热力图已保存: {output_path}")
    print(f"  尺寸: 75行 × 68列 (高75, 宽68)")
    
    # 保存数据
    data_path = os.path.join(output_dir, "F2_heatmap_data_75x68_v7.csv")
    pd.DataFrame(final_heatmap).to_csv(data_path)
    print(f"✓ 数据已保存: {data_path}")
    
    print("\n" + "=" * 60)
    print("完成！热力图尺寸: 75行 × 68列 (高75, 宽68)")
    print("水平白线分隔5部分 (N | P1 | P2 | S1 | S2)，垂直虚线分隔4单元")
    print("=" * 60)

if __name__ == "__main__":
    main()
