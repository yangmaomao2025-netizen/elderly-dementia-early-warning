#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨被试单情绪对比分析
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

def analyze_cross_subject(analysis_name, emotion, subject_files, output_dir):
    """
    分析多个被试在单一情绪下的对比
    subject_files: {subject_id: csv_file_path}
    """
    
    print(f"\n{'='*60}")
    print(f"分析: {analysis_name}")
    print(f"情绪: {emotion}")
    print(f"被试数: {len(subject_files)}")
    print(f"{'='*60}")
    
    # 创建输出目录
    for subdir in ['statistics', 'heatmaps', 'barplots', 'boxplots']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # 读取所有被试数据
    subject_data = {}
    all_au_cols = None
    
    for subject, file_path in subject_files.items():
        df = pd.read_csv(file_path, on_bad_lines='skip')
        subject_data[subject] = df
        if all_au_cols is None:
            all_au_cols = [c for c in df.columns if c.endswith('_r')]
        print(f"  {subject}: {len(df)} 帧")
    
    au_labels = [c.replace('_r', '').replace(' ', '') for c in all_au_cols]
    subjects = list(subject_files.keys())
    
    # 1. 计算各被试AU均值
    subject_means = []
    for subject, df in subject_data.items():
        row = {'subject': subject, 'n_frames': len(df)}
        for au in all_au_cols:
            row[au.strip()] = df[au].mean()
        subject_means.append(row)
    
    means_df = pd.DataFrame(subject_means)
    means_df.to_csv(os.path.join(output_dir, 'statistics', 'subject_au_means.csv'), index=False)
    
    # 2. 统计描述
    stats_summary = []
    for au_clean in [c.strip() for c in all_au_cols]:
        values = means_df[au_clean].values
        stats_summary.append({
            'AU': au_clean,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'cv': np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        })
    
    stats_df = pd.DataFrame(stats_summary)
    stats_df = stats_df.sort_values('cv', ascending=False)
    stats_df.to_csv(os.path.join(output_dir, 'statistics', 'au_variability.csv'), index=False)
    
    print(f"\n变异系数最高的5个AU:")
    for _, row in stats_df.head(5).iterrows():
        print(f"  {row['AU']}: CV={row['cv']:.3f}, mean={row['mean']:.3f}")
    
    # 3. 被试×AU热力图
    fig, ax = plt.subplots(figsize=(16, 6))
    heatmap_data = means_df[[c.strip() for c in all_au_cols]].values
    
    sns.heatmap(heatmap_data,
               xticklabels=au_labels,
               yticklabels=means_df['subject'].values,
               annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
               cbar_kws={'label': 'Mean Intensity'})
    ax.set_title(f'{analysis_name} - Subject × AU Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmaps', 'heatmap_subject_au.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 各AU被试对比柱状图
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, au_clean in enumerate([c.strip() for c in all_au_cols[:17]]):
        ax = axes[i]
        values = means_df[au_clean].values
        colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
        bars = ax.bar(means_df['subject'].values, values, color=colors, alpha=0.8)
        ax.set_title(au_clean, fontsize=10, fontweight='bold')
        ax.set_ylabel('Mean Intensity')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    plt.suptitle(f'{analysis_name} - AU Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'barplots', 'barplot_au_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. 箱线图 (合并所有被试数据)
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, au in enumerate(all_au_cols[:17]):
        ax = axes[i]
        data_to_plot = [subject_data[s][au].values for s in subjects]
        bp = ax.boxplot(data_to_plot, labels=subjects, patch_artist=True)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(au.strip(), fontsize=10, fontweight='bold')
        ax.set_ylabel('Intensity')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'{analysis_name} - AU Distribution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplots', 'boxplot_au_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. 生成报告
    report = f"""================================================================================
跨被试单情绪对比分析报告
分析名称: {analysis_name}
情绪类型: {emotion}
分析日期: {datetime.now().strftime('%Y-%m-%d')}
================================================================================

1. 被试信息
-----------
被试列表: {', '.join(subjects)}
被试数量: {len(subjects)}
总帧数: {sum([len(df) for df in subject_data.values()])}

2. AU变异情况
-------------
变异系数 (CV) 统计:
  平均CV: {stats_df['cv'].mean():.3f}
  最大CV: {stats_df['cv'].max():.3f} ({stats_df.loc[stats_df['cv'].idxmax(), 'AU']})
  最小CV: {stats_df['cv'].min():.3f} ({stats_df.loc[stats_df['cv'].idxmin(), 'AU']})

Top 5 变异最大AU:
"""
    
    for i, row in stats_df.head(5).iterrows():
        report += f"  {i+1}. {row['AU']}: CV={row['cv']:.3f}, range=[{row['min']:.2f}, {row['max']:.2f}]\n"
    
    report += f"""
3. 组内一致性
-------------
一致性评估: {'高' if stats_df['cv'].mean() < 0.3 else '中' if stats_df['cv'].mean() < 0.5 else '低'}
平均变异系数: {stats_df['cv'].mean():.3f}

4. 输出文件
-----------
statistics/
  - subject_au_means.csv
  - au_variability.csv

heatmaps/
  - heatmap_subject_au.png

barplots/
  - barplot_au_comparison.png

boxplots/
  - boxplot_au_distribution.png

================================================================================
"""
    
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 完成: {output_dir}")

# ============ 定义12个分析任务 ============

analyses = [
    # 患者组
    ("患者组_积极情绪对比", "积极", {
        'ZFL': '2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_积极.csv',
        'MHD': '../sj/MHD_积极.csv',
        'WGL': '../sj/WGL_积极.csv',
        'ZJK': '../sj/ZJK_积极.csv'
    }),
    ("患者组_悲伤情绪对比", "悲伤", {
        'ZFL': '2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_悲伤.csv',
        'MHD': '../sj/MHD_悲伤.csv',
        'WGL': '../sj/WGL_悲伤.csv',
        'ZJK': '../sj/ZJK_悲伤.csv'
    }),
    ("患者组_中性情绪对比", "中性", {
        'ZFL': '2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_中性.csv',
        'MHD': '../sj/MHD_中性.csv',
        'WGL': '../sj/WGL_中性.csv',
        'ZJK': '../sj/ZJK_中性.csv'
    }),
    
    # 正常组
    ("正常组_积极情绪对比", "积极", {
        'M1': '../zcsj/M1_积极.csv',
        'M2': '../zcsj/M2_积极.csv',
        'F1': '../zcsj/F1_积极.csv',
        'F2': '../zcsj/F2_积极.csv'
    }),
    ("正常组_悲伤情绪对比", "悲伤", {
        'M1': '../zcsj/M1_悲伤.csv',
        'M2': '../zcsj/M2_悲伤.csv',
        'F1': '../zcsj/F1_悲伤.csv',
        'F2': '../zcsj/F2_悲伤.csv'
    }),
    ("正常组_中性情绪对比", "中性", {
        'M1': '../zcsj/M1_中性.csv',
        'M2': '../zcsj/M2_中性.csv',
        'F1': '../zcsj/F1_中性.csv',
        'F2': '../zcsj/F2_中性.csv'
    }),
    
    # 跨组男性 (M1, M2 + ZFL, WGL, ZJK)
    ("跨组男性_积极情绪对比", "积极", {
        'M1_正常': '../zcsj/M1_积极.csv',
        'M2_正常': '../zcsj/M2_积极.csv',
        'ZFL_患者': '2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_积极.csv',
        'WGL_患者': '../sj/WGL_积极.csv',
        'ZJK_患者': '../sj/ZJK_积极.csv'
    }),
    ("跨组男性_悲伤情绪对比", "悲伤", {
        'M1_正常': '../zcsj/M1_悲伤.csv',
        'M2_正常': '../zcsj/M2_悲伤.csv',
        'ZFL_患者': '2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_悲伤.csv',
        'WGL_患者': '../sj/WGL_悲伤.csv',
        'ZJK_患者': '../sj/ZJK_悲伤.csv'
    }),
    ("跨组男性_中性情绪对比", "中性", {
        'M1_正常': '../zcsj/M1_中性.csv',
        'M2_正常': '../zcsj/M2_中性.csv',
        'ZFL_患者': '2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_中性.csv',
        'WGL_患者': '../sj/WGL_中性.csv',
        'ZJK_患者': '../sj/ZJK_中性.csv'
    }),
    
    # 跨组女性 (F1, F2 + MHD)
    ("跨组女性_积极情绪对比", "积极", {
        'F1_正常': '../zcsj/F1_积极.csv',
        'F2_正常': '../zcsj/F2_积极.csv',
        'MHD_患者': '../sj/MHD_积极.csv'
    }),
    ("跨组女性_悲伤情绪对比", "悲伤", {
        'F1_正常': '../zcsj/F1_悲伤.csv',
        'F2_正常': '../zcsj/F2_悲伤.csv',
        'MHD_患者': '../sj/MHD_悲伤.csv'
    }),
    ("跨组女性_中性情绪对比", "中性", {
        'F1_正常': '../zcsj/F1_中性.csv',
        'F2_正常': '../zcsj/F2_中性.csv',
        'MHD_患者': '../sj/MHD_中性.csv'
    }),
]

if __name__ == '__main__':
    print("="*60)
    print("开始12个跨被试情绪对比分析")
    print("="*60)
    
    for i, (name, emotion, files) in enumerate(analyses, 1):
        output_dir = f"2025-02-23_{name}"
        print(f"\n[{i}/12] ", end="")
        analyze_cross_subject(name, emotion, files, output_dir)
    
    print("\n" + "="*60)
    print("✓ 全部12个分析完成!")
    print("="*60)
