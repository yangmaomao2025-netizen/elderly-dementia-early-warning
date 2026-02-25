#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量生成75×68 AU热力图应用程序
Batch Heatmap Generator for AU Analysis

用法：
    python heatmap_generator.py --input <输入文件夹> --output <输出文件夹> [--label <组别标签>]

示例：
    python heatmap_generator.py --input ./zcsj --output ./output --label "正常组"
    python heatmap_generator.py --input ./sj --output ./output --label "患者组"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import glob
import argparse
from datetime import datetime

# matplotlib配置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# AU强度列名
AU_R_COLS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
             'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
             'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

# AU标签
AU_LABELS = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 
             'AU09', 'AU10', 'AU12', 'AU14', 'AU15', 'AU17', 
             'AU20', 'AU23', 'AU25', 'AU26', 'AU45']

# Y轴部分标签
PART_LABELS = ['N\n(Neutral)', 'P1\n(Positive 0-60s)', 'P2\n(Positive 60-120s)', 
               'S1\n(Sad 61-120s)', 'S2\n(Sad 121-180s)']
PART_CENTERS = [7.5, 22.5, 37.5, 52.5, 67.5]


def preprocess_data(csv_file, start_sec, end_sec):
    """
    数据预处理：
    1. 剔除success为0的行
    2. 保留指定秒数范围
    3. 对连续1秒的AU强度值求和
    """
    try:
        df = pd.read_csv(csv_file, on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        # 剔除success为0的行
        df = df[df['success'] == 1].copy()
        
        # 时间戳归零
        df['timestamp'] = df['timestamp'] - df['timestamp'].min()
        
        # 保留指定范围
        df = df[(df['timestamp'] >= start_sec) & (df['timestamp'] < end_sec)].copy()
        df['timestamp'] = df['timestamp'] - df['timestamp'].min()
        df['second'] = df['timestamp'].astype(int)
        
        # AU列
        au_cols_clean = [col.strip() for col in AU_R_COLS]
        au_sums = df.groupby('second')[au_cols_clean].sum()
        
        # 确保所有秒数都有数据
        target_seconds = end_sec - start_sec
        all_seconds = pd.DataFrame({'second': range(target_seconds)})
        au_sums = all_seconds.merge(au_sums.reset_index(), on='second', how='left').fillna(0)
        au_sums = au_sums.drop('second', axis=1)
        
        return au_sums
    except Exception as e:
        print(f"    错误: {e}")
        return None


def create_unit_transposed(au_sums, start_sec, end_sec):
    """创建单元：15行(秒) × 17列(AU)"""
    return au_sums.iloc[start_sec:end_sec].values


def generate_x_tick_labels():
    """生成X轴刻度标签：AU{编号}_U{单元编号}"""
    labels = []
    for unit in range(1, 5):
        for au in AU_LABELS:
            labels.append(f"{au}_U{unit}")
    return labels


def generate_heatmap(base_dir, subject_id, output_dir, group_label=""):
    """为单个被试生成热力图"""
    
    # 检查文件
    neutral_file = os.path.join(base_dir, f"{subject_id}_中性.csv")
    positive_file = os.path.join(base_dir, f"{subject_id}_积极.csv")
    sad_file = os.path.join(base_dir, f"{subject_id}_悲伤.csv")
    
    missing_files = [f for f in [neutral_file, positive_file, sad_file] if not os.path.exists(f)]
    if missing_files:
        print(f"  ✗ 缺少文件: {missing_files}")
        return False
    
    print(f"  处理: {subject_id}...", end=" ")
    
    # 数据预处理
    neutral = preprocess_data(neutral_file, 0, 60)
    positive = preprocess_data(positive_file, 0, 120)
    sad = preprocess_data(sad_file, 61, 181)
    
    if neutral is None or positive is None or sad is None:
        print("失败")
        return False
    
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
    
    # 绘制
    fig, ax = plt.subplots(figsize=(24, 16))
    sns.heatmap(final_heatmap, cmap='RdBu_r', center=0, 
                xticklabels=False, yticklabels=False,
                cbar_kws={'label': 'AU Intensity Sum'}, ax=ax)
    
    # X轴刻度
    x_tick_labels = generate_x_tick_labels()
    ax.set_xticks([i + 0.5 for i in range(68)])
    ax.set_xticklabels(x_tick_labels, fontsize=6, rotation=90)
    
    # Y轴刻度
    ax.set_yticks(PART_CENTERS)
    ax.set_yticklabels(PART_LABELS, fontsize=10, rotation=0)
    
    # 标题
    title_suffix = f" - {group_label}" if group_label else ""
    ax.set_title(f'{subject_id} AU Heatmap (75 × 68){title_suffix}\nHeight: 75 rows | Width: 68 cols', 
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
    
    print("✓")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='批量生成75×68 AU热力图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python heatmap_generator.py --input ./zcsj --output ./output
  python heatmap_generator.py --input ./sj --output ./output --label "患者组"
  python heatmap_generator.py -i ./data -o ./results -l "Control Group"
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='输入文件夹路径')
    parser.add_argument('-o', '--output', required=True, help='输出文件夹路径')
    parser.add_argument('-l', '--label', default='', help='组别标签（可选）')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 检查输入文件夹
    if not os.path.exists(args.input):
        print(f"错误: 输入文件夹不存在: {args.input}")
        sys.exit(1)
    
    # 创建输出文件夹
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("批量生成75×68 AU热力图")
    print("=" * 60)
    print(f"输入文件夹: {args.input}")
    print(f"输出文件夹: {args.output}")
    if args.label:
        print(f"组别标签: {args.label}")
    print()
    
    # 扫描被试ID
    csv_files = glob.glob(os.path.join(args.input, "*_中性.csv"))
    subjects = [os.path.basename(f).replace("_中性.csv", "") for f in csv_files]
    subjects = sorted(set(subjects))
    
    if not subjects:
        print("错误: 未找到符合命名规范的文件（*_中性.csv）")
        sys.exit(1)
    
    print(f"发现被试: {', '.join(subjects)}")
    print(f"共 {len(subjects)} 个被试")
    print()
    
    # 批量生成
    success_count = 0
    start_time = datetime.now()
    
    for subject_id in subjects:
        if generate_heatmap(args.input, subject_id, args.output, args.label):
            success_count += 1
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print()
    print("=" * 60)
    print(f"完成！成功生成 {success_count}/{len(subjects)} 个热力图")
    print(f"耗时: {duration:.1f} 秒")
    print(f"输出目录: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
