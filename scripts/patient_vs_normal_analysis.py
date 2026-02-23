#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
患者组 vs 正常对照组 直接统计检验
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

print("="*60)
print("患者组 vs 正常对照组 统计检验")
print("="*60)

output_dir = "2025-02-23_患者组vs正常组_统计检验"
os.makedirs(f"{output_dir}/statistics", exist_ok=True)
os.makedirs(f"{output_dir}/heatmaps", exist_ok=True)
os.makedirs(f"{output_dir}/barplots", exist_ok=True)
os.makedirs(f"{output_dir}/boxplots", exist_ok=True)
os.makedirs(f"{output_dir}/volcano", exist_ok=True)

# 定义被试文件
patient_subjects = {
    'ZFL': {
        '悲伤': '2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_悲伤.csv',
        '积极': '2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_积极.csv',
        '中性': '2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_中性.csv'
    },
    'MHD': {
        '悲伤': '../sj/MHD_悲伤.csv',
        '积极': '../sj/MHD_积极.csv',
        '中性': '../sj/MHD_中性.csv'
    },
    'WGL': {
        '悲伤': '../sj/WGL_悲伤.csv',
        '积极': '../sj/WGL_积极.csv',
        '中性': '../sj/WGL_中性.csv'
    },
    'ZJK': {
        '悲伤': '../sj/ZJK_悲伤.csv',
        '积极': '../sj/ZJK_积极.csv',
        '中性': '../sj/ZJK_中性.csv'
    }
}

normal_subjects = {
    'M1': {
        '悲伤': '../zcsj/M1_悲伤.csv',
        '积极': '../zcsj/M1_积极.csv',
        '中性': '../zcsj/M1_中性.csv'
    },
    'M2': {
        '悲伤': '../zcsj/M2_悲伤.csv',
        '积极': '../zcsj/M2_积极.csv',
        '中性': '../zcsj/M2_中性.csv'
    },
    'F1': {
        '悲伤': '../zcsj/F1_悲伤.csv',
        '积极': '../zcsj/F1_积极.csv',
        '中性': '../zcsj/F1_中性.csv'
    },
    'F2': {
        '悲伤': '../zcsj/F2_悲伤.csv',
        '积极': '../zcsj/F2_积极.csv',
        '中性': '../zcsj/F2_中性.csv'
    }
}

emotions = ['悲伤', '积极', '中性']

# 计算各被试-情绪的AU均值
def calculate_subject_means(subjects_dict, group_name):
    """计算被试均值"""
    results = []
    for subject, emotion_files in subjects_dict.items():
        for emotion, file_path in emotion_files.items():
            df = pd.read_csv(file_path, on_bad_lines='skip')
            au_cols = [c for c in df.columns if c.endswith('_r')]
            
            row = {'subject': subject, 'group': group_name, 'emotion': emotion, 'n_frames': len(df)}
            for au in au_cols:
                row[au.strip()] = df[au].mean()
            results.append(row)
    return pd.DataFrame(results), au_cols

print("\n计算被试AU均值...")
patient_df, au_cols = calculate_subject_means(patient_subjects, '患者组')
normal_df, _ = calculate_subject_means(normal_subjects, '正常对照组')

all_df = pd.concat([patient_df, normal_df], ignore_index=True)
all_df.to_csv(f'{output_dir}/statistics/all_subject_emotion_au_means.csv', index=False)

print(f"患者组: {len(patient_df)} 条记录")
print(f"正常对照组: {len(normal_df)} 条记录")
print(f"AU指标数: {len(au_cols)}")

au_labels = [c.replace('_r', '').replace(' ', '') for c in au_cols]

# 按情绪进行组间统计检验
print("\n进行组间统计检验...")
comparison_results = []

for emotion in emotions:
    print(f"\n  {emotion}情绪:")
    
    emotion_patient = patient_df[patient_df['emotion'] == emotion]
    emotion_normal = normal_df[normal_df['emotion'] == emotion]
    
    for au_clean in [c.strip() for c in au_cols]:
        patient_values = emotion_patient[au_clean].values
        normal_values = emotion_normal[au_clean].values
        
        # t-test
        t_stat, p_value = stats.ttest_ind(patient_values, normal_values)
        
        # Cohen's d (效应量)
        pooled_std = np.sqrt(((len(patient_values)-1)*np.var(patient_values, ddof=1) + 
                             (len(normal_values)-1)*np.var(normal_values, ddof=1)) / 
                             (len(patient_values) + len(normal_values) - 2))
        cohens_d = (np.mean(patient_values) - np.mean(normal_values)) / pooled_std if pooled_std > 0 else 0
        
        comparison_results.append({
            'emotion': emotion,
            'AU': au_clean,
            'patient_mean': np.mean(patient_values),
            'patient_std': np.std(patient_values),
            'normal_mean': np.mean(normal_values),
            'normal_std': np.std(normal_values),
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        })

results_df = pd.DataFrame(comparison_results)
results_df.to_csv(f'{output_dir}/statistics/group_comparison_statistics.csv', index=False)

# 筛选显著差异的AU
sig_results = results_df[results_df['significant']].copy()
sig_results = sig_results.sort_values('p_value')

print(f"\n显著差异AU总数: {len(sig_results)} / {len(results_df)}")

# 各情绪显著差异AU数
for emotion in emotions:
    n_sig = len(sig_results[sig_results['emotion'] == emotion])
    print(f"  {emotion}: {n_sig} 个AU显著差异")

# Top 10 最显著差异
print("\nTop 10 最显著差异:")
for _, row in sig_results.head(10).iterrows():
    effect = "患者>正常" if row['cohens_d'] > 0 else "患者<正常"
    print(f"  {row['emotion']}-{row['AU']}: p={row['p_value']:.2e}, d={row['cohens_d']:.2f} ({effect})")

# 1. 火山图 (Volcano Plot) - 按情绪
print("\n生成火山图...")
for emotion in emotions:
    emotion_results = results_df[results_df['emotion'] == emotion].copy()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 计算 -log10(p)
    emotion_results['neg_log_p'] = -np.log10(emotion_results['p_value'].clip(lower=1e-300))
    
    # 显著性着色
    colors = []
    for _, row in emotion_results.iterrows():
        if row['p_value'] < 0.05 and row['cohens_d'] > 0.5:
            colors.append('#e74c3c')  # 红色: 患者高
        elif row['p_value'] < 0.05 and row['cohens_d'] < -0.5:
            colors.append('#3498db')  # 蓝色: 正常高
        else:
            colors.append('#95a5a6')  # 灰色: 不显著
    
    scatter = ax.scatter(emotion_results['cohens_d'], emotion_results['neg_log_p'], 
                        c=colors, alpha=0.7, s=80, edgecolors='white', linewidth=0.5)
    
    # 添加显著性阈值线
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5, label='p=0.05')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.3)
    
    # 标注Top差异AU
    top_sig = emotion_results[emotion_results['significant']].nsmallest(5, 'p_value')
    for _, row in top_sig.iterrows():
        ax.annotate(row['AU'], (row['cohens_d'], row['neg_log_p']), 
                   fontsize=9, ha='center', fontweight='bold')
    
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    ax.set_title(f'{emotion} - Patient vs Normal Volcano Plot', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/volcano/volcano_{emotion}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {emotion} 火山图")

# 2. 组间差异热力图
print("\n生成组间差异热力图...")
for emotion in emotions:
    emotion_results = results_df[results_df['emotion'] == emotion]
    
    # 准备差异数据
    diff_data = emotion_results['cohens_d'].values.reshape(1, -1)
    
    fig, ax = plt.subplots(figsize=(16, 3))
    sns.heatmap(diff_data, xticklabels=au_labels, yticklabels=['Cohen\'s d'],
               annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
               cbar_kws={'label': 'Effect Size (Patient - Normal)'})
    ax.set_title(f'{emotion} - Patient vs Normal Effect Size', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmaps/heatmap_effect_size_{emotion}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {emotion} 效应量热力图")

# 3. 显著差异AU柱状图
print("\n生成显著差异AU柱状图...")
for emotion in emotions:
    emotion_sig = sig_results[sig_results['emotion'] == emotion].head(10)
    
    if len(emotion_sig) == 0:
        continue
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(emotion_sig))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, emotion_sig['patient_mean'], width, label='患者组', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, emotion_sig['normal_mean'], width, label='正常对照组', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title(f'{emotion} - Significantly Different AUs (Top 10)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_sig['AU'].values, rotation=45, ha='right')
    ax.legend()
    
    # 添加p值标注
    for i, (_, row) in enumerate(emotion_sig.iterrows()):
        sig_marker = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*'
        ax.text(i, max(row['patient_mean'], row['normal_mean']) * 1.05, sig_marker, 
               ha='center', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/barplots/barplot_significant_au_{emotion}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {emotion} 显著差异AU图")

# 4. 箱线图对比
print("\n生成箱线图对比...")
for emotion in emotions:
    emotion_df = all_df[all_df['emotion'] == emotion]
    
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, au_clean in enumerate([c.strip() for c in au_cols[:17]]):
        ax = axes[i]
        
        patient_vals = emotion_df[emotion_df['group'] == '患者组'][au_clean].values
        normal_vals = emotion_df[emotion_df['group'] == '正常对照组'][au_clean].values
        
        bp = ax.boxplot([patient_vals, normal_vals], labels=['患者组', '正常组'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#3498db')
        bp['boxes'][1].set_alpha(0.7)
        
        ax.set_title(au_clean, fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean Intensity')
    
    plt.suptitle(f'{emotion} - Patient vs Normal Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplots/boxplot_group_comparison_{emotion}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {emotion} 箱线图")

# 5. 生成综合报告
print("\n生成综合报告...")

report = f"""================================================================================
患者组 vs 正常对照组 统计检验报告
分析日期: {datetime.now().strftime('%Y-%m-%d')}
================================================================================

1. 样本信息
-----------
患者组被试: ZFL, MHD, WGL, ZJK (n=4)
正常对照组: M1, M2, F1, F2 (n=4)
分析AU数: {len(au_cols)}

2. 统计方法
-----------
组间比较: 独立样本t检验
效应量: Cohen's d
显著性水平: α = 0.05

3. 显著差异汇总
---------------
总显著差异AU数: {len(sig_results)} / {len(results_df)}

按情绪分类:
  悲伤: {len(sig_results[sig_results['emotion']=='悲伤'])} 个AU显著差异
  积极: {len(sig_results[sig_results['emotion']=='积极'])} 个AU显著差异
  中性: {len(sig_results[sig_results['emotion']=='中性'])} 个AU显著差异

4. Top 显著差异AU
-----------------
"""

for emotion in emotions:
    report += f"\n{emotion}情绪:\n"
    emotion_sig = sig_results[sig_results['emotion'] == emotion].head(5)
    for i, (_, row) in enumerate(emotion_sig.iterrows(), 1):
        direction = "患者组更高" if row['cohens_d'] > 0 else "正常组更高"
        report += f"  {i}. {row['AU']}: p={row['p_value']:.2e}, d={row['cohens_d']:.2f} ({direction})\n"

report += """
5. 主要发现
-----------
"""

# 患者组显著更高的AU
patient_higher = sig_results[sig_results['cohens_d'] > 0.5]
if len(patient_higher) > 0:
    report += "患者组显著高表达的AU:\n"
    for emotion in emotions:
        emotion_higher = patient_higher[patient_higher['emotion'] == emotion]
        if len(emotion_higher) > 0:
            aus = ', '.join(emotion_higher.head(3)['AU'].values)
            report += f"  {emotion}: {aus}\n"

# 正常组显著更高的AU
normal_higher = sig_results[sig_results['cohens_d'] < -0.5]
if len(normal_higher) > 0:
    report += "\n正常组显著高表达的AU:\n"
    for emotion in emotions:
        emotion_higher = normal_higher[normal_higher['emotion'] == emotion]
        if len(emotion_higher) > 0:
            aus = ', '.join(emotion_higher.head(3)['AU'].values)
            report += f"  {emotion}: {aus}\n"

report += f"""
6. 输出文件清单
---------------
statistics/
  - all_subject_emotion_au_means.csv
  - group_comparison_statistics.csv

heatmaps/
  - heatmap_effect_size_悲伤.png
  - heatmap_effect_size_积极.png
  - heatmap_effect_size_中性.png

barplots/
  - barplot_significant_au_悲伤.png
  - barplot_significant_au_积极.png
  - barplot_significant_au_中性.png

boxplots/
  - boxplot_group_comparison_悲伤.png
  - boxplot_group_comparison_积极.png
  - boxplot_group_comparison_中性.png

volcano/
  - volcano_悲伤.png
  - volcano_积极.png
  - volcano_中性.png

================================================================================
"""

with open(f'{output_dir}/analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✓ 患者组vs正常组分析完成: {output_dir}")
print(f"  总分析数: {len(results_df)}")
print(f"  显著差异: {len(sig_results)}")
