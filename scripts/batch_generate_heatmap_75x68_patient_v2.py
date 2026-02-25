#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成75×68 AU热力图 - 患者组 (sj/)
X轴刻度：AU{编号}_U{单元编号}
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# AU强度列名
AU_R_COLS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
             'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
             'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

# AU标签（不带_r后缀）
AU_LABELS = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 
             'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 
             'AU20', 'AU23', 'AU25', 'AU26', 'AU45']

def preprocess_data(csv_file, start_sec, end_sec):
    """数据预处理"""
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    df = df[df['success'] == 1].copy()
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    df = df[(df['timestamp'] >= start_sec) & (df['timestamp'] < end_sec)].copy()
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    df['second'] = df['timestamp'].astype(int)
    au_cols_clean = [col.strip() for col in AU_R_COLS]
    au_sums = df.groupby('second')[au_cols_clean].sum()
    target_seconds = end_sec - start_sec
    all_seconds = pd.DataFrame({'second': range(target_seconds)})
    au_sums = all_seconds.merge(au_sums.reset_index(), on='second', how='left').fillna(0)
    au_sums = au_sums.drop('second', axis=1)
    return au_sums

def create_unit_transposed(au_sums, start_sec, end_sec):
    """创建单元：15×17"""
    return au_sums.iloc[start_sec:end_sec].values

def generate_heatmap_for_subject(base_dir, subject_id, output_dir):
    """为单个被试生成热力图"""
    print(f"\n处理患者: {subject_id}")
    
    # 检查文件是否存在
    neutral_file = os.path.join(base_dir, f"{subject_id}_中性.csv")
    positive_file = os.path.join(base_dir, f"{subject_id}_积极.csv")
    sad_file = os.path.join(base_dir, f"{subject_id}_悲伤.csv")
    
    for f in [neutral_file, positive_file, sad_file]:
        if not os.path.exists(f):
            print(f"  ✗ 文件不存在: {f}")
            return False
    
    print(f"  ✓ 找到所有文件")
    
    # 数据预处理
    neutral = preprocess_data(neutral_file, 0, 60)
    positive = preprocess_data(positive_file, 0, 120)
    sad = preprocess_data(sad_file, 61, 181)
    
    # 生成各部分热力图
    parts = []
    
    # 中性
    units = [create_unit_transposed(neutral, i*15, (i+1)*15) for i in range(4)]
    parts.append(np.hstack(units))
    
    # 积极前60秒
    units = [create_unit_transposed(positive.iloc[:60], i*15, (i+1)*15) for i in range(4)]
    parts.append(np.hstack(units))
    
    # 积极后60秒
    units = [create_unit_transposed(positive.iloc[60:120], i*15, (i+1)*15) for i in range(4)]
    parts.append(np.hstack(units))
    
    # 悲伤前60秒
    units = [create_unit_transposed(sad.iloc[:60], i*15, (i+1)*15) for i in range(4)]
    parts.append(np.hstack(units))
    
    # 悲伤后60秒
    units = [create_unit_transposed(sad.iloc[60:120], i*15, (i+1)*15) for i in range(4)]
    parts.append(np.hstack(units))
    
    # 拼接
    final_heatmap = np.vstack(parts)
    print(f"  热力图尺寸: {final_heatmap.shape}")
    
    # 绘制
    fig, ax = plt.subplots(figsize=(24, 16))
    sns.heatmap(final_heatmap, cmap='RdBu_r', center=0, 
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'AU Intensity Sum'}, ax=ax)
    
    # X轴刻度：每个刻度显示 AU{编号}_U{单元编号}
    x_tick_positions = list(range(68))  # 0到67
    x_tick_labels = []
    for unit in range(1, 5):
        for au in AU_LABELS:
            x_tick_labels.append(f"{au}_U{unit}")
    
    ax.set_xticks([i + 0.5 for i in x_tick_positions])  # 刻度在单元格中心
    ax.set_xticklabels(x_tick_labels, fontsize=6, rotation=90)
    
    # Y轴刻度
    part_centers = [7.5, 22.5, 37.5, 52.5, 67.5]
    part_labels = ['N\n(Neutral)', 'P1\n(Positive 0-60s)', 'P2\n(Positive 60-120s)', 
                   'S1\n(Sad 61-120s)', 'S2\n(Sad 121-180s)']
    ax.set_yticks(part_centers)
    ax.set_yticklabels(part_labels, fontsize=10, rotation=0)
    
    ax.set_title(f'{subject_id} AU Heatmap (75 × 68) - Patient\nHeight: 75 rows | Width: 68 cols', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Width: 68 cols (4 units × 17 AU)', fontsize=10)
    ax.set_ylabel('Height: 75 rows (N | P1 | P2 | S1 | S2)', fontsize=10)
    
    # 分割线
    for y in [14.5, 29.5, 44.5, 59.5]:
        ax.axhline(y=y, color='white', linewidth=2)
    for x in [16.5, 33.5, 50.5]:
        ax.axvline(x=x, color='white', linewidth=1, linestyle='--')
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, f"{subject_id}_heatmap_75x68.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    data_path = os.path.join(output_dir, f"{subject_id}_heatmap_data_75x68.csv")
    pd.DataFrame(final_heatmap).to_csv(data_path)
    
    print(f"  ✓ 已保存: {output_path}")
    return True

def main():
    base_dir = "/root/.openclaw/workspace/老年失智人群预警模式科研项目/sj"
    output_dir = "/root/.openclaw/workspace/老年失智人群预警模式科研项目/analysis_results/批量热力图_75x68_患者组"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("批量生成75×68 AU热力图 - 患者组")
    print("="*60)
    print(f"输入文件夹: {base_dir}")
    print(f"输出文件夹: {output_dir}")
    
    # 扫描被试ID
    csv_files = glob.glob(os.path.join(base_dir, "*_中性.csv"))
    subjects = [os.path.basename(f).replace("_中性.csv", "") for f in csv_files]
    subjects = sorted(set(subjects))
    
    print(f"\n发现患者: {', '.join(subjects)}")
    print(f"共 {len(subjects)} 个患者")
    
    # 批量生成
    success_count = 0
    for subject_id in subjects:
        if generate_heatmap_for_subject(base_dir, subject_id, output_dir):
            success_count += 1
    
    print("\n" + "="*60)
    print(f"完成！成功生成 {success_count}/{len(subjects)} 个热力图")
    print(f"输出目录: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
