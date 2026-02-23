#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨被试组对比分析脚本
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

def analyze_group(group_name, subject_files, output_dir):
    """
    分析一组被试的跨被试对比
    subject_files: {subject_id: (sad_file, pos_file, neu_file)}
    """
    
    print(f"\n{'='*60}")
    print(f"分析组: {group_name}")
    print(f"被试数: {len(subject_files)}")
    print(f"{'='*60}")
    
    # 创建输出目录
    subdirs = ['raw_data', 'heatmaps', 'barplots', 'boxplots', 'statistics']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # 读取所有被试数据
    all_data = []
    for subject_id, (sad_f, pos_f, neu_f) in subject_files.items():
        sad_df = pd.read_csv(sad_f, on_bad_lines='skip')
        pos_df = pd.read_csv(pos_f, on_bad_lines='skip')
        neu_df = pd.read_csv(neu_f, on_bad_lines='skip')
        
        sad_df['emotion'] = '悲伤'
        pos_df['emotion'] = '积极'
        neu_df['emotion'] = '中性'
        sad_df['subject'] = subject_id
        pos_df['subject'] = subject_id
        neu_df['subject'] = subject_id
        
        all_data.extend([sad_df, pos_df, neu_df])
    
    all_df = pd.concat(all_data, ignore_index=True)
    
    # 获取AU列
    sample_df = all_data[0]
    au_r_cols = [col for col in sample_df.columns if col.endswith('_r')]
    au_labels = [col.replace('_r', '').replace(' ', '') for col in au_r_cols]
    
    print(f"AU指标数: {len(au_r_cols)}")
    print(f"总样本量: {len(all_df)} 帧")
    
    subjects = list(subject_files.keys())
    emotions = ['悲伤', '积极', '中性']
    
    # 1. 计算每个被试-情绪组合的AU均值
    subject_emotion_stats = []
    for subject in subjects:
        for emotion in emotions:
            subset = all_df[(all_df['subject']==subject) & (all_df['emotion']==emotion)]
            if len(subset) == 0:
                continue
            row = {'subject': subject, 'emotion': emotion, 'n_frames': len(subset)}
            for au in au_r_cols:
                row[au.strip()] = subset[au].mean()
            subject_emotion_stats.append(row)
    
    stats_df = pd.DataFrame(subject_emotion_stats)
    stats_df.to_csv(os.path.join(output_dir, 'statistics', 'subject_emotion_au_means.csv'), index=False)
    
    # 2. 跨被试情绪对比 (各情绪内的被试差异)
    emotion_subject_comparison = []
    for emotion in emotions:
        emotion_data = stats_df[stats_df['emotion']==emotion]
        for au in [col.strip() for col in au_r_cols]:
            values = emotion_data[au].values
            emotion_subject_comparison.append({
                'emotion': emotion,
                'AU': au,
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'cv': np.std(values)/np.mean(values) if np.mean(values) > 0 else 0
            })
    
    comparison_df = pd.DataFrame(emotion_subject_comparison)
    comparison_df.to_csv(os.path.join(output_dir, 'statistics', 'emotion_subject_variability.csv'), index=False)
    
    # 3. 创建热力图 (被试 × AU)
    for emotion in emotions:
        fig, ax = plt.subplots(figsize=(16, 6))
        emotion_data = stats_df[stats_df['emotion']==emotion]
        au_cols_clean = [col.strip() for col in au_r_cols]
        heatmap_data = emotion_data[au_cols_clean].values
        
        sns.heatmap(heatmap_data, 
                   xticklabels=au_labels,
                   yticklabels=emotion_data['subject'].values,
                   annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                   cbar_kws={'label': 'Mean Intensity'})
        ax.set_title(f'{emotion} - Subject × AU Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmaps', f'heatmap_subject_au_{emotion}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. 三情绪对比热力图 (平均)
    fig, ax = plt.subplots(figsize=(16, 6))
    avg_by_emotion = []
    for emotion in emotions:
        emotion_data = stats_df[stats_df['emotion']==emotion]
        row = [emotion_data[au].mean() for au in [col.strip() for col in au_r_cols]]
        avg_by_emotion.append(row)
    
    heatmap_df = pd.DataFrame(avg_by_emotion, columns=au_labels, index=emotions)
    sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
               cbar_kws={'label': 'Mean Intensity'})
    ax.set_title(f'{group_name} - Average AU by Emotion', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmaps', 'heatmap_group_emotion_comparison.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. 被试间变异系数热力图
    fig, ax = plt.subplots(figsize=(16, 6))
    cv_data = []
    for emotion in emotions:
        emotion_stats = comparison_df[comparison_df['emotion']==emotion]
        cv_row = emotion_stats['cv'].values
        cv_data.append(cv_row)
    
    cv_df = pd.DataFrame(cv_data, columns=au_labels, index=emotions)
    sns.heatmap(cv_df, annot=True, fmt='.3f', cmap='Reds', ax=ax,
               cbar_kws={'label': 'Coefficient of Variation'})
    ax.set_title(f'{group_name} - Inter-Subject Variability (CV)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmaps', 'heatmap_inter_subject_cv.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. 各情绪被试对比柱状图
    for emotion in emotions:
        fig, axes = plt.subplots(3, 6, figsize=(20, 10))
        axes = axes.flatten()
        
        emotion_data = stats_df[stats_df['emotion']==emotion]
        for i, au in enumerate([col.strip() for col in au_r_cols[:17]]):
            ax = axes[i]
            values = emotion_data[au].values
            bars = ax.bar(emotion_data['subject'].values, values, 
                         color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(values)], alpha=0.8)
            ax.set_title(au, fontsize=10, fontweight='bold')
            ax.set_ylabel('Mean Intensity')
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'{emotion} - Subject Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'barplots', f'barplot_subject_comparison_{emotion}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # 7. 生成报告
    generate_group_report(group_name, subjects, stats_df, comparison_df, output_dir)
    
    print(f"✓ 组分析完成: {output_dir}")
    return stats_df

def generate_group_report(group_name, subjects, stats_df, comparison_df, output_dir):
    """生成组分析报告"""
    
    report = f"""================================================================================
跨被试组分析报告
组名: {group_name}
被试: {', '.join(subjects)}
日期: {datetime.now().strftime('%Y-%m-%d')}
================================================================================

1. 数据概况
-----------
被试数量: {len(subjects)}
被试列表: {', '.join(subjects)}
情绪类型: 悲伤、积极、中性

2. 各情绪组内被试差异
---------------------
"""
    
    for emotion in ['悲伤', '积极', '中性']:
        emotion_comp = comparison_df[comparison_df['emotion']==emotion]
        report += f"\n{emotion}情绪:\n"
        report += f"  平均变异系数: {emotion_comp['cv'].mean():.3f}\n"
        report += f"  变异最大AU: {emotion_comp.loc[emotion_comp['cv'].idxmax(), 'AU']} (CV={emotion_comp['cv'].max():.3f})\n"
        report += f"  变异最小AU: {emotion_comp.loc[emotion_comp['cv'].idxmin(), 'AU']} (CV={emotion_comp['cv'].min():.3f})\n"
    
    report += """
3. 组内一致性评估
-----------------
"""
    
    # 计算组内一致性 (低CV表示高一致性)
    avg_cv_by_emotion = comparison_df.groupby('emotion')['cv'].mean()
    for emotion, cv in avg_cv_by_emotion.items():
        consistency = "高" if cv < 0.3 else "中" if cv < 0.5 else "低"
        report += f"  {emotion}情绪一致性: {consistency} (平均CV={cv:.3f})\n"
    
    report += """
4. 输出文件清单
---------------
heatmaps/
  - heatmap_subject_au_悲伤.png
  - heatmap_subject_au_积极.png
  - heatmap_subject_au_中性.png
  - heatmap_group_emotion_comparison.png
  - heatmap_inter_subject_cv.png

barplots/
  - barplot_subject_comparison_悲伤.png
  - barplot_subject_comparison_积极.png
  - barplot_subject_comparison_中性.png

statistics/
  - subject_emotion_au_means.csv
  - emotion_subject_variability.csv

================================================================================
"""
    
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == '__main__':
    import sys
    import json
    
    if len(sys.argv) < 4:
        print("用法: python3 analyze_group.py <group_name> <output_dir> <subject:files_json>")
        sys.exit(1)
    
    group_name = sys.argv[1]
    output_dir = sys.argv[2]
    subjects_json = sys.argv[3]
    
    subject_files = json.loads(subjects_json)
    analyze_group(group_name, subject_files, output_dir)
