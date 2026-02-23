#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三情绪AU特征分析脚本
参考: 2025-02-17_M1_三种情绪对比
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

# 设置绘图参数
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

def analyze_subject(subject_id, sad_file, pos_file, neu_file, output_dir):
    """分析单个被试的三情绪数据"""
    
    print(f"\n{'='*60}")
    print(f"分析被试: {subject_id}")
    print(f"{'='*60}")
    
    # 创建输出目录
    subdirs = ['raw_data', 'heatmaps', 'barplots', 'boxplots', 'radar', 'time_series', 'statistics']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # 读取数据 (跳过格式错误行)
    sad_df = pd.read_csv(sad_file, on_bad_lines='skip')
    pos_df = pd.read_csv(pos_file, on_bad_lines='skip')
    neu_df = pd.read_csv(neu_file, on_bad_lines='skip')
    
    # 添加情绪标签
    sad_df['emotion'] = '悲伤'
    pos_df['emotion'] = '积极'
    neu_df['emotion'] = '中性'
    
    # 保存原始数据
    sad_df.to_csv(os.path.join(output_dir, 'raw_data', f'{subject_id}_悲伤.csv'), index=False)
    pos_df.to_csv(os.path.join(output_dir, 'raw_data', f'{subject_id}_积极.csv'), index=False)
    neu_df.to_csv(os.path.join(output_dir, 'raw_data', f'{subject_id}_中性.csv'), index=False)
    
    # 获取AU列
    au_r_cols = [col for col in sad_df.columns if col.endswith('_r')]
    au_labels = [col.replace('_r', '').replace(' ', '') for col in au_r_cols]
    
    print(f"样本量: 悲伤={len(sad_df)}, 积极={len(pos_df)}, 中性={len(neu_df)}")
    print(f"AU指标数: {len(au_r_cols)}")
    
    # 合并数据
    all_df = pd.concat([sad_df, pos_df, neu_df], ignore_index=True)
    
    # 1. 描述性统计
    stats_results = []
    for au in au_r_cols:
        sad_vals = sad_df[au].values
        pos_vals = pos_df[au].values
        neu_vals = neu_df[au].values
        
        # ANOVA
        f_stat, p_anova = stats.f_oneway(sad_vals, pos_vals, neu_vals)
        
        stats_results.append({
            'AU': au.strip(),
            '悲伤_mean': np.mean(sad_vals),
            '积极_mean': np.mean(pos_vals),
            '中性_mean': np.mean(neu_vals),
            '悲伤_std': np.std(sad_vals),
            '积极_std': np.std(pos_vals),
            '中性_std': np.std(neu_vals),
            'F_stat': f_stat,
            'p_value': p_anova
        })
    
    stats_df = pd.DataFrame(stats_results)
    stats_df = stats_df.sort_values('F_stat', ascending=False)
    stats_df.to_csv(os.path.join(output_dir, 'statistics', 'au_emotion_statistics.csv'), index=False)
    
    print(f"\nTop 5 最具区分度AU (F值):")
    for i, row in stats_df.head(5).iterrows():
        print(f"  {row['AU']}: F={row['F_stat']:.1f}, p={row['p_value']:.2e}")
    
    # 2. 热力图
    create_heatmaps(sad_df, pos_df, neu_df, au_r_cols, au_labels, output_dir)
    
    # 3. 柱状图
    create_barplots(stats_df, output_dir)
    
    # 4. 箱线图 (Top 6 AU)
    create_boxplots(all_df, stats_df.head(6)['AU'].values, output_dir)
    
    # 5. 雷达图
    create_radar(stats_df, output_dir)
    
    # 6. 时间序列 (Top 3 AU)
    create_timeseries(sad_df, pos_df, neu_df, stats_df.head(3)['AU'].values, output_dir)
    
    # 7. 生成报告
    generate_report(subject_id, stats_df, len(sad_df), len(pos_df), len(neu_df), output_dir)
    
    print(f"✓ 分析完成: {output_dir}")
    return stats_df

def create_heatmaps(sad_df, pos_df, neu_df, au_cols, au_labels, output_dir):
    """创建热力图"""
    emotions = {'悲伤': sad_df, '积极': pos_df, '中性': neu_df}
    
    # 单个情绪热力图
    for emotion_name, df in emotions.items():
        fig, ax = plt.subplots(figsize=(12, 8))
        data = df[au_cols].mean().values.reshape(1, -1)
        sns.heatmap(data, xticklabels=au_labels, yticklabels=[emotion_name], 
                   annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Intensity'})
        ax.set_title(f'{emotion_name} - AU Intensity Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmaps', f'heatmap_{emotion_name}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 三情绪对比热力图
    fig, ax = plt.subplots(figsize=(14, 6))
    data = pd.DataFrame({
        '悲伤': sad_df[au_cols].mean().values,
        '积极': pos_df[au_cols].mean().values,
        '中性': neu_df[au_cols].mean().values
    }, index=au_labels)
    sns.heatmap(data, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Intensity'})
    ax.set_title('Three Emotions AU Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmaps', 'heatmap_all_emotions_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 悲伤vs积极差异热力图
    fig, ax = plt.subplots(figsize=(14, 2))
    diff = sad_df[au_cols].mean().values - pos_df[au_cols].mean().values
    sns.heatmap(diff.reshape(1, -1), xticklabels=au_labels, yticklabels=['Sad - Positive'],
               annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax, cbar_kws={'label': 'Difference'})
    ax.set_title('Sad vs Positive AU Difference', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmaps', 'heatmap_sad_vs_positive_diff.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_barplots(stats_df, output_dir):
    """创建柱状图"""
    # 所有AU均值对比
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(stats_df))
    width = 0.25
    
    ax.bar(x - width, stats_df['悲伤_mean'], width, label='Sad', color='#3498db', alpha=0.8)
    ax.bar(x, stats_df['积极_mean'], width, label='Positive', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, stats_df['中性_mean'], width, label='Neutral', color='#95a5a6', alpha=0.8)
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title('AU Mean Intensity Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['AU'], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'barplots', 'barplot_au_mean_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Top 8 AU对比
    fig, ax = plt.subplots(figsize=(14, 6))
    top8 = stats_df.head(8)
    x = np.arange(len(top8))
    
    ax.bar(x - width, top8['悲伤_mean'], width, label='Sad', color='#3498db', alpha=0.8)
    ax.bar(x, top8['积极_mean'], width, label='Positive', color='#e74c3c', alpha=0.8)
    ax.bar(x + width, top8['中性_mean'], width, label='Neutral', color='#95a5a6', alpha=0.8)
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title('Top 8 AU Mean Intensity Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(top8['AU'], rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'barplots', 'barplot_key_au_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_boxplots(all_df, top_aus, output_dir):
    """创建箱线图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    # 清理AU名称（去除空格）
    clean_aus = [au.strip() for au in top_aus]
    
    for i, au_clean in enumerate(clean_aus[:6]):
        ax = axes[i]
        # 找到原始列名
        orig_col = None
        for col in all_df.columns:
            if col.strip() == au_clean:
                orig_col = col
                break
        if orig_col is None:
            continue
            
        data = [all_df[all_df['emotion']=='悲伤'][orig_col].values,
                all_df[all_df['emotion']=='积极'][orig_col].values,
                all_df[all_df['emotion']=='中性'][orig_col].values]
        bp = ax.boxplot(data, labels=['Sad', 'Positive', 'Neutral'], patch_artist=True)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')
        bp['boxes'][2].set_facecolor('#95a5a6')
        ax.set_title(au_clean, fontsize=11, fontweight='bold')
        ax.set_ylabel('Intensity')
    
    plt.suptitle('Key AU Distribution Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplots', 'boxplot_key_au_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_radar(stats_df, output_dir):
    """创建雷达图"""
    top6 = stats_df.head(6)
    categories = [au.strip() for au in top6['AU'].values]
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    for emotion, color in [('悲伤_mean', '#3498db'), ('积极_mean', '#e74c3c'), ('中性_mean', '#95a5a6')]:
        values = top6[emotion].values.tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=emotion.replace('_mean', ''), color=color)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Emotion Profile Radar (Top 6 AU)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar', 'radar_emotion_profile.png'), dpi=150, bbox_inches='tight')
    plt.close()

def create_timeseries(sad_df, pos_df, neu_df, top_aus, output_dir):
    """创建时间序列图"""
    for au_clean in [au.strip() for au in top_aus]:
        # 找到原始列名
        orig_col = None
        for col in sad_df.columns:
            if col.strip() == au_clean:
                orig_col = col
                break
        if orig_col is None:
            continue
            
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        
        for ax, (df, emotion, color) in zip(axes, [
            (sad_df, 'Sad', '#3498db'),
            (pos_df, 'Positive', '#e74c3c'),
            (neu_df, 'Neutral', '#95a5a6')
        ]):
            x_vals = df[' timestamp'].values if ' timestamp' in df.columns else range(len(df))
            ax.plot(x_vals, df[orig_col].values, color=color, alpha=0.7, linewidth=0.8)
            ax.set_ylabel(emotion, fontsize=11)
            ax.set_ylim(0, max(df[orig_col].max() * 1.1, 0.1))
            ax.grid(True, alpha=0.3)
        
        axes[0].set_title(f'{au_clean} Time Series', fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Frame', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_series', f'timeseries_{au_clean}.png'), dpi=150, bbox_inches='tight')
        plt.close()

def generate_report(subject_id, stats_df, sad_n, pos_n, neu_n, output_dir):
    """生成分析报告"""
    top5 = stats_df.head(5)
    
    report = f"""================================================================================
三情绪AU特征分析报告
被试ID: {subject_id}
日期: {datetime.now().strftime('%Y-%m-%d')}
================================================================================

1. 数据概况
-----------
情绪类型: 悲伤、积极、中性
数据来源: OpenFace 2.0 AU强度值

样本量:
  • 悲伤: {sad_n} 帧
  • 积极: {pos_n} 帧  
  • 中性: {neu_n} 帧

2. 核心发现
-----------
最具区分度的AU (按F值排序):
"""
    
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        sig = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else ''
        report += f"  {i}. {row['AU']:10s}: F={row['F_stat']:8.1f}, p={row['p_value']:.2e} {sig}\n"
    
    report += """
3. 情绪特异性模式
-----------------
"""
    
    # 悲伤特征
    sad_aus = top5[top5['悲伤_mean'] > top5[['悲伤_mean', '积极_mean', '中性_mean']].max(axis=1)]['AU'].values
    if len(sad_aus) > 0:
        report += f"悲伤情绪特征:\n"
        for au in sad_aus[:3]:
            report += f"  • {au}: 最高激活\n"
    
    # 积极特征
    pos_aus = top5[top5['积极_mean'] > top5[['悲伤_mean', '积极_mean', '中性_mean']].max(axis=1)]['AU'].values
    if len(pos_aus) > 0:
        report += f"\n积极情绪特征:\n"
        for au in pos_aus[:3]:
            report += f"  • {au}: 最高激活\n"
    
    # 中性特征
    report += f"\n中性情绪特征:\n"
    report += f"  • 所有AU强度普遍较低\n"
    
    report += """
4. 输出文件清单
---------------
heatmaps/
  - heatmap_悲伤.png
  - heatmap_积极.png
  - heatmap_中性.png
  - heatmap_all_emotions_comparison.png
  - heatmap_sad_vs_positive_diff.png

barplots/
  - barplot_au_mean_comparison.png
  - barplot_key_au_comparison.png

boxplots/
  - boxplot_key_au_distribution.png

radar/
  - radar_emotion_profile.png

time_series/
  - timeseries_[Top3_AU].png

statistics/
  - au_emotion_statistics.csv

================================================================================
"""
    
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 6:
        print("用法: python3 analyze_emotions.py <subject_id> <sad_csv> <pos_csv> <neu_csv> <output_dir>")
        print(f"当前参数数量: {len(sys.argv)-1}")
        sys.exit(1)
    
    subject_id, sad_file, pos_file, neu_file, output_dir = sys.argv[1:6]
    analyze_subject(subject_id, sad_file, pos_file, neu_file, output_dir)
