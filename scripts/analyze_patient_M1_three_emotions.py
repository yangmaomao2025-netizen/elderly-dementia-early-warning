#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
患者M1三种情绪对比分析
分析M1在悲伤、中性、积极三种情绪下的AU特征差异
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

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# M1患者三种情绪文件
emotion_files = {
    '悲伤': '/root/.openclaw/media/inbound/file_27---2b92310a-c235-4003-a321-a0ec90aae5ba.csv',
    '中性': '/root/.openclaw/media/inbound/file_30---022dee2d-6990-4df3-aa41-a71b974bb24d.csv',
    '积极': '/root/.openclaw/media/inbound/file_24---925f9a2e-ba59-4283-829c-75d596785181.csv'
}

au_columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

def load_and_process(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def analyze_m1_cross_emotion():
    print("="*70)
    print("患者M1三种情绪对比分析")
    print("="*70)
    
    emotion_data = {}
    for emotion, filepath in emotion_files.items():
        try:
            df = load_and_process(filepath)
            emotion_data[emotion] = df
            print(f"{emotion}: {len(df)} frames")
        except Exception as e:
            print(f"Error loading {emotion}: {e}")
    
    results = []
    for emotion, df in emotion_data.items():
        au_means = df[au_columns].mean()
        au_means['emotion'] = emotion
        results.append(au_means)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('emotion')
    
    print("\n各情绪AU均值:")
    print(results_df.round(3))
    
    return emotion_data, results_df

def create_barplots(emotion_data, results_df, output_dir):
    barplots_dir = Path(output_dir) / 'barplots'
    barplots_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(au_columns))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, emotion in enumerate(['悲伤', '中性', '积极']):
        values = results_df.loc[emotion].values
        ax.bar(x + idx*width, values, width, label=emotion, color=colors[idx])
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('激活强度', fontsize=12)
    ax.set_title('患者M1 - 三种情绪AU均值对比', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([au.replace('_r', '') for au in au_columns], rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(barplots_dir / 'barplot_au_mean_comparison.png', dpi=150)
    plt.close()
    
    key_aus = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU17_r']
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(key_aus))
    
    for idx, emotion in enumerate(['悲伤', '中性', '积极']):
        values = [results_df.loc[emotion, au] for au in key_aus]
        ax.bar(x + idx*width, values, width, label=emotion, color=colors[idx])
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('激活强度', fontsize=12)
    ax.set_title('患者M1 - 关键AU三情绪对比', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([au.replace('_r', '') for au in key_aus])
    ax.legend()
    plt.tight_layout()
    plt.savefig(barplots_dir / 'barplot_key_au_comparison.png', dpi=150)
    plt.close()

def create_boxplots(emotion_data, output_dir):
    boxplots_dir = Path(output_dir) / 'boxplots'
    boxplots_dir.mkdir(exist_ok=True)
    
    key_aus = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU17_r']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, au in enumerate(key_aus):
        data_to_plot = []
        labels = []
        for emotion in ['悲伤', '中性', '积极']:
            data_to_plot.append(emotion_data[emotion][au].values)
            labels.append(emotion)
        
        axes[idx].boxplot(data_to_plot, labels=labels)
        axes[idx].set_title(f'{au.replace("_r", "")} 分布')
        axes[idx].set_ylabel('激活强度')
    
    for idx in range(len(key_aus), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('患者M1 - 关键AU分布箱线图', fontsize=16)
    plt.tight_layout()
    plt.savefig(boxplots_dir / 'boxplot_key_au_distribution.png', dpi=150)
    plt.close()

def create_heatmaps(emotion_data, results_df, output_dir):
    heatmaps_dir = Path(output_dir) / 'heatmaps'
    heatmaps_dir.mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(results_df.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, ax=ax, cbar_kws={'label': 'AU Activation'})
    ax.set_title('患者M1 - 三种情绪AU激活热图', fontsize=14)
    ax.set_xlabel('情绪类型')
    ax.set_ylabel('Action Units')
    plt.tight_layout()
    plt.savefig(heatmaps_dir / 'heatmap_all_emotions_comparison.png', dpi=150)
    plt.close()
    
    diff_sad_pos = results_df.loc['悲伤'] - results_df.loc['积极']
    fig, ax = plt.subplots(figsize=(14, 2))
    diff_df = pd.DataFrame(diff_sad_pos).T
    diff_df.index = ['悲伤-积极']
    sns.heatmap(diff_df, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, ax=ax, cbar_kws={'label': '差异值'})
    ax.set_title('患者M1 - 悲伤vs积极AU差异', fontsize=12)
    plt.tight_layout()
    plt.savefig(heatmaps_dir / 'heatmap_sad_vs_positive_diff.png', dpi=150)
    plt.close()
    
    for emotion in ['悲伤', '中性', '积极']:
        fig, ax = plt.subplots(figsize=(14, 2))
        emotion_df = pd.DataFrame(results_df.loc[emotion]).T
        emotion_df.index = [emotion]
        sns.heatmap(emotion_df, annot=True, fmt='.2f', cmap='RdYlBu_r',
                    center=0, ax=ax, cbar_kws={'label': '激活值'})
        ax.set_title(f'患者M1 - {emotion}情绪AU激活', fontsize=12)
        plt.tight_layout()
        plt.savefig(heatmaps_dir / f'heatmap_{emotion}.png', dpi=150)
        plt.close()

def create_radar(emotion_data, results_df, output_dir):
    radar_dir = Path(output_dir) / 'radar'
    radar_dir.mkdir(exist_ok=True)
    
    key_aus = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU17_r']
    angles = np.linspace(0, 2*np.pi, len(key_aus), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, emotion in enumerate(['悲伤', '中性', '积极']):
        values = [results_df.loc[emotion, au] for au in key_aus]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=emotion, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([au.replace('_r', '') for au in key_aus])
    ax.set_ylim(0, results_df[key_aus].max().max() * 1.2)
    ax.set_title('患者M1 - 三种情绪AU轮廓雷达图', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(radar_dir / 'radar_emotion_profile.png', dpi=150)
    plt.close()

def create_time_series(emotion_data, output_dir):
    ts_dir = Path(output_dir) / 'time_series'
    ts_dir.mkdir(exist_ok=True)
    
    key_aus = ['AU04_r', 'AU06_r', 'AU12_r']
    
    for au in key_aus:
        fig, ax = plt.subplots(figsize=(12, 5))
        for emotion in ['悲伤', '中性', '积极']:
            data = emotion_data[emotion][au].values
            target_len = min(1000, len(data))
            indices = np.linspace(0, len(data)-1, target_len, dtype=int)
            data_downsampled = data[indices]
            time_sec = np.arange(len(data_downsampled)) * 0.033
            ax.plot(time_sec, data_downsampled, label=emotion, alpha=0.7)
        
        ax.set_xlabel('时间 (秒)', fontsize=12)
        ax.set_ylabel(f'{au.replace("_r", "")} 激活强度', fontsize=12)
        ax.set_title(f'患者M1 - {au.replace("_r", "")}时间序列对比', fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.savefig(ts_dir / f'timeseries_{au}.png', dpi=150)
        plt.close()

def save_raw_data(emotion_data, output_dir):
    raw_dir = Path(output_dir) / 'raw_data'
    raw_dir.mkdir(exist_ok=True)
    
    for emotion, df in emotion_data.items():
        key_cols = ['frame', 'timestamp', 'confidence'] + au_columns
        df[key_cols].to_csv(raw_dir / f'{emotion}.csv', index=False, encoding='utf-8-sig')

def create_statistics(emotion_data, results_df, output_dir):
    stats_dir = Path(output_dir) / 'statistics'
    stats_dir.mkdir(exist_ok=True)
    
    f_values = []
    for au in au_columns:
        sad_data = emotion_data['悲伤'][au].values
        neu_data = emotion_data['中性'][au].values
        pos_data = emotion_data['积极'][au].values
        
        f_stat, p_val = stats.f_oneway(sad_data, neu_data, pos_data)
        f_values.append({'AU': au, 'F_value': f_stat, 'p_value': p_val})
    
    f_df = pd.DataFrame(f_values)
    f_df = f_df.sort_values('F_value', ascending=False)
    f_df.to_csv(stats_dir / 'au_emotion_statistics.csv', index=False, encoding='utf-8-sig')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'gray' 
              for p in f_df['p_value']]
    bars = ax.bar(range(len(f_df)), f_df['F_value'], color=colors)
    ax.set_xticks(range(len(f_df)))
    ax.set_xticklabels([au.replace('_r', '') for au in f_df['AU']], rotation=45)
    ax.set_ylabel('F值', fontsize=12)
    ax.set_title('患者M1 - 三情绪ANOVA F值 (红色:p<0.001, 橙色:p<0.01)', fontsize=14)
    plt.tight_layout()
    plt.savefig(stats_dir / 'statistics_anova_f_values.png', dpi=150)
    plt.close()

def generate_report(emotion_data, results_df, output_dir):
    output_path = Path(output_dir) / 'analysis_report.txt'
    
    f_values = []
    for au in au_columns:
        sad_data = emotion_data['悲伤'][au].values
        neu_data = emotion_data['中性'][au].values
        pos_data = emotion_data['积极'][au].values
        f_stat, p_val = stats.f_oneway(sad_data, neu_data, pos_data)
        f_values.append((au, f_stat, p_val))
    
    f_values.sort(key=lambda x: x[1], reverse=True)
    
    report = f"""================================================================================
患者M1三种情绪AU特征分析报告
日期: {pd.Timestamp.now().strftime('%Y-%m-%d')}
================================================================================

1. 数据概况
-----------
患者: 男性抑郁症患者 M1
情绪类型: 悲伤、中性、积极
数据来源: OpenFace 2.0 AU强度值

样本量:
  • 悲伤: {len(emotion_data['悲伤'])} 帧
  • 中性: {len(emotion_data['中性'])} 帧
  • 积极: {len(emotion_data['积极'])} 帧

2. 核心发现
-----------
最具区分度的AU (按F值排序):
"""
    
    for au, f_val, p_val in f_values[:10]:
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        report += f"  {au:10s}: F={f_val:8.1f}, p={p_val:.2e} {sig}\n"
    
    report += f"""
3. 各情绪AU均值
---------------
"""
    for emotion in ['悲伤', '中性', '积极']:
        report += f"\n{emotion}:\n"
        for au in ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU17_r']:
            report += f"  {au}: {results_df.loc[emotion, au]:.3f}\n"
    
    report += """
4. 结论
-------
患者M1在三种情绪下表现出特定的AU激活模式。
可用于评估患者对情绪刺激的表达反应。

================================================================================
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存: {output_path}")

def main():
    output_dir = Path('/root/.openclaw/workspace/老年失智人群预警模式科研项目/analysis_results/2026-02-18_患者M1_三种情绪对比')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("患者M1三种情绪对比分析")
    print("="*70)
    
    emotion_data, results_df = analyze_m1_cross_emotion()
    
    print("\n生成条形图...")
    create_barplots(emotion_data, results_df, output_dir)
    
    print("生成箱线图...")
    create_boxplots(emotion_data, output_dir)
    
    print("生成热图...")
    create_heatmaps(emotion_data, results_df, output_dir)
    
    print("生成雷达图...")
    create_radar(emotion_data, results_df, output_dir)
    
    print("生成时间序列图...")
    create_time_series(emotion_data, output_dir)
    
    print("保存原始数据...")
    save_raw_data(emotion_data, output_dir)
    
    print("生成统计分析...")
    create_statistics(emotion_data, results_df, output_dir)
    
    print("生成分析报告...")
    generate_report(emotion_data, results_df, output_dir)
    
    print(f"\n{'='*70}")
    print(f"分析完成！结果保存在: {output_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
