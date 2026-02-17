#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
患者-对照组悲伤情绪对比分析（修正版）
使用正确的文件映射：
- 男患者: file_27, file_28, file_29
- 女患者: file_39, file_40, file_41
- 对照组: file_3, file_4, file_5
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 正确的文件映射
file_mapping = {
    '对照组': {
        'M1': '/root/.openclaw/media/inbound/file_3---b3314058-964d-470d-8293-13430fdde2c6.csv',
        'M2': '/root/.openclaw/media/inbound/file_4---0dd96eb3-72ff-4ced-a1b8-c5c51fad721a.csv',
        'F1': '/root/.openclaw/media/inbound/file_5---69ad20a2-5a2f-4f18-bdef-056d8c24d515.csv'
    },
    '男患者': {
        'M1': '/root/.openclaw/media/inbound/file_27---2b92310a-c235-4003-a321-a0ec90aae5ba.csv',
        'M2': '/root/.openclaw/media/inbound/file_28---a22ed529-d93d-4a5d-8bc6-2415bf19cbfe.csv',
        'M3': '/root/.openclaw/media/inbound/file_29---ee23776c-8da6-44c4-aba3-8a74eca4f174.csv'
    },
    '女患者': {
        'F1': '/root/.openclaw/media/inbound/file_39---a156b700-0dc5-4a7d-98a7-842a4d9ee1da.csv',
        'F2': '/root/.openclaw/media/inbound/file_40---571ec375-dfd5-403f-b0da-0922d9ff79e6.csv',
        'F3': '/root/.openclaw/media/inbound/file_41---90b39ad4-e124-4e19-84ad-d2beb4ddcc4e.csv'
    }
}

au_columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def cohen_d(x, y):
    """计算Cohen's d效应量"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*x.var() + (ny-1)*y.var()) / dof)
    return (x.mean() - y.mean()) / pooled_std if pooled_std > 0 else 0

def analyze_group(group_name, files_dict):
    """分析一个组的AU数据"""
    print(f"\n{'='*60}")
    print(f"分析 {group_name}")
    print(f"{'='*60}")
    
    all_data = []
    subject_stats = {}
    
    for subject_id, filepath in files_dict.items():
        df = load_data(filepath)
        mean_values = df[au_columns].mean()
        all_data.append(mean_values)
        subject_stats[subject_id] = mean_values
        print(f"{subject_id}: {len(df)} 帧")
    
    # 计算组内平均
    group_mean = pd.DataFrame(all_data).mean()
    group_std = pd.DataFrame(all_data).std()
    
    print(f"\n{group_name} AU激活均值 (Top 5):")
    sorted_aus = group_mean.sort_values(ascending=False)
    for au in sorted_aus.head(5).index:
        print(f"  {au}: {group_mean[au]:.3f} ± {group_std[au]:.3f}")
    
    return group_mean, group_std, subject_stats

def compare_groups(control_mean, patient_mean_male, patient_mean_female, output_dir):
    """对比对照组和患者组"""
    print(f"\n{'='*60}")
    print("对照组 vs 患者组对比")
    print(f"{'='*60}")
    
    # 合并患者数据（男女合并）
    patient_combined = pd.concat([patient_mean_male, patient_mean_female], axis=1).mean(axis=1)
    
    # 计算差异
    diff_control_vs_patient = patient_combined - control_mean
    
    print("\n对照组 vs 合并患者组 - 最大差异AU:")
    sorted_diff = diff_control_vs_patient.sort_values(key=abs, ascending=False)
    for au in sorted_diff.head(10).index:
        direction = "↑" if diff_control_vs_patient[au] > 0 else "↓"
        print(f"  {au}: {control_mean[au]:.3f} → {patient_combined[au]:.3f} ({diff_control_vs_patient[au]:+.3f}) {direction}")
    
    # 性别特异性差异
    print("\n\n男性患者 vs 对照组 - 最大差异AU:")
    diff_male = patient_mean_male - control_mean
    sorted_diff_male = diff_male.sort_values(key=abs, ascending=False)
    for au in sorted_diff_male.head(5).index:
        print(f"  {au}: {control_mean[au]:.3f} → {patient_mean_male[au]:.3f} ({diff_male[au]:+.3f})")
    
    print("\n女性患者 vs 对照组 - 最大差异AU:")
    diff_female = patient_mean_female - control_mean
    sorted_diff_female = diff_female.sort_values(key=abs, ascending=False)
    for au in sorted_diff_female.head(5).index:
        print(f"  {au}: {control_mean[au]:.3f} → {patient_mean_female[au]:.3f} ({diff_female[au]:+.3f})")
    
    # 保存对比结果
    comparison_df = pd.DataFrame({
        '对照组': control_mean,
        '男患者': patient_mean_male,
        '女患者': patient_mean_female,
        '患者合并': patient_combined,
        '男患者-对照': diff_male,
        '女患者-对照': diff_female
    })
    
    comparison_df.to_csv(output_dir / 'statistics' / 'patient_control_comparison.csv')
    print(f"\n对比结果已保存")
    
    return comparison_df

def create_comparison_plots(control_mean, patient_mean_male, patient_mean_female, output_dir):
    """创建对比图表"""
    # 1. 热图对比
    fig, ax = plt.subplots(figsize=(16, 4))
    
    comparison_data = pd.DataFrame({
        '对照组': control_mean,
        '男患者': patient_mean_male,
        '女患者': patient_mean_female
    })
    
    sns.heatmap(comparison_data.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, ax=ax, cbar_kws={'label': 'AU Activation'})
    ax.set_title('悲伤情绪 - 对照组 vs 患者组 AU激活对比', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmaps' / 'patient_control_comparison_heatmap.png', dpi=150)
    plt.close()
    print("热图已保存")
    
    # 2. 差异条形图
    fig, ax = plt.subplots(figsize=(14, 6))
    
    diff_male = patient_mean_male - control_mean
    diff_female = patient_mean_female - control_mean
    
    x = np.arange(len(au_columns))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, diff_male.values, width, label='男患者 vs 对照', alpha=0.8)
    bars2 = ax.bar(x + width/2, diff_female.values, width, label='女患者 vs 对照', alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Difference (患者 - 对照)', fontsize=12)
    ax.set_title('悲伤情绪 - 患者组vs对照组 AU激活差异', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([au.replace('_r', '') for au in au_columns], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'barplots' / 'patient_control_difference_bars.png', dpi=150)
    plt.close()
    print("差异条形图已保存")

def main():
    # 设置输出目录
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d')
    base_dir = Path(f'/root/.openclaw/workspace/analysis_results/{timestamp}_患者对照组对比_悲伤情绪_修正版')
    
    # 创建子目录
    for subdir in ['heatmaps', 'barplots', 'boxplots', 'statistics']:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("患者-对照组悲伤情绪对比分析（修正版）")
    print("="*70)
    print("\n正确文件映射:")
    print("  对照组: file_3 (M1), file_4 (M2), file_5 (F1)")
    print("  男患者: file_27 (M1), file_28 (M2), file_29 (M3)")
    print("  女患者: file_39 (F1), file_40 (F2), file_41 (F3)")
    
    # 分析各组
    control_mean, control_std, control_subjects = analyze_group('对照组', file_mapping['对照组'])
    patient_male_mean, patient_male_std, patient_male_subjects = analyze_group('男患者', file_mapping['男患者'])
    patient_female_mean, patient_female_std, patient_female_subjects = analyze_group('女患者', file_mapping['女患者'])
    
    # 组间对比
    comparison_df = compare_groups(control_mean, patient_male_mean, patient_female_mean, base_dir)
    
    # 创建图表
    create_comparison_plots(control_mean, patient_male_mean, patient_female_mean, base_dir)
    
    # 生成报告
    report = f"""# 患者-对照组悲伤情绪对比分析报告（修正版）

## 数据说明
- 对照组: 2男 + 1女 (file_3, file_4, file_5)
- 男患者: 3男 (file_27, file_28, file_29)
- 女患者: 3女 (file_39, file_40, file_41)
- 情绪: 悲伤

## 关键发现

### 对照组平均AU激活 (Top 5)
{control_mean.sort_values(ascending=False).head().to_string()}

### 男患者平均AU激活 (Top 5)
{patient_male_mean.sort_values(ascending=False).head().to_string()}

### 女患者平均AU激活 (Top 5)
{patient_female_mean.sort_values(ascending=False).head().to_string()}

### 主要差异AU
{comparison_df[['男患者-对照', '女患者-对照']].sort_values(by='男患者-对照', key=abs, ascending=False).head(10).to_string()}
"""
    
    with open(base_dir / 'statistics' / 'analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n{'='*70}")
    print(f"分析完成！结果保存在: {base_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
