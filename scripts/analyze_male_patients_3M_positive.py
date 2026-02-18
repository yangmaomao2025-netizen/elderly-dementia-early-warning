#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
男性患者3M积极情绪对比分析
对比M1/M2/M3在积极情绪下的AU特征差异
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

# 男性患者积极情绪文件
patient_files = {
    'M1': '/root/.openclaw/media/inbound/file_24---925f9a2e-ba59-4283-829c-75d596785181.csv',
    'M2': '/root/.openclaw/media/inbound/file_25---f701a00a-5efc-44e6-8514-4510879be7a9.csv',
    'M3': '/root/.openclaw/media/inbound/file_26---2b859f5a-08e2-4713-b654-c56162c1085d.csv'
}

au_columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

def load_and_process(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def analyze_patient_positive():
    print("="*70)
    print("男性患者3M积极情绪对比分析")
    print("="*70)
    
    patient_data = {}
    for pid, filepath in patient_files.items():
        try:
            df = load_and_process(filepath)
            patient_data[pid] = df
            print(f"{pid}: {len(df)} frames")
        except Exception as e:
            print(f"Error loading {pid}: {e}")
    
    # 计算各患者的AU均值
    results = []
    for pid, df in patient_data.items():
        au_means = df[au_columns].mean()
        au_means['patient'] = pid
        results.append(au_means)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('patient')
    
    print("\n各患者AU均值:")
    print(results_df.round(3))
    
    # 计算患者间差异
    print("\n患者间差异分析:")
    print(f"M1 vs M2: {(results_df.loc['M1'] - results_df.loc['M2']).abs().mean():.3f}")
    print(f"M1 vs M3: {(results_df.loc['M1'] - results_df.loc['M3']).abs().mean():.3f}")
    print(f"M2 vs M3: {(results_df.loc['M2'] - results_df.loc['M3']).abs().mean():.3f}")
    
    return patient_data, results_df

def create_barplots(patient_data, results_df, output_dir):
    barplots_dir = Path(output_dir) / 'barplots'
    barplots_dir.mkdir(exist_ok=True)
    
    # 1. AU均值对比条形图
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(au_columns))
    width = 0.25
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, pid in enumerate(['M1', 'M2', 'M3']):
        values = results_df.loc[pid].values
        ax.bar(x + idx*width, values, width, label=pid, color=colors[idx])
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('激活强度', fontsize=12)
    ax.set_title('男性患者3M - 积极情绪AU均值对比', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([au.replace('_r', '') for au in au_columns], rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig(barplots_dir / 'barplot_au_mean_comparison.png', dpi=150)
    plt.close()
    
    # 2. 关键AU对比条形图
    key_aus = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU17_r']
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(key_aus))
    
    for idx, pid in enumerate(['M1', 'M2', 'M3']):
        values = [results_df.loc[pid, au] for au in key_aus]
        ax.bar(x + idx*width, values, width, label=pid, color=colors[idx])
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('激活强度', fontsize=12)
    ax.set_title('男性患者3M - 积极情绪关键AU对比', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels([au.replace('_r', '') for au in key_aus])
    ax.legend()
    plt.tight_layout()
    plt.savefig(barplots_dir / 'barplot_key_au_comparison.png', dpi=150)
    plt.close()

def create_boxplots(patient_data, output_dir):
    boxplots_dir = Path(output_dir) / 'boxplots'
    boxplots_dir.mkdir(exist_ok=True)
    
    key_aus = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU17_r']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, au in enumerate(key_aus):
        data_to_plot = []
        labels = []
        for pid in ['M1', 'M2', 'M3']:
            data_to_plot.append(patient_data[pid][au].values)
            labels.append(pid)
        
        axes[idx].boxplot(data_to_plot, labels=labels)
        axes[idx].set_title(f'{au.replace("_r", "")} 分布')
        axes[idx].set_ylabel('激活强度')
    
    for idx in range(len(key_aus), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('男性患者3M - 积极情绪关键AU分布箱线图', fontsize=16)
    plt.tight_layout()
    plt.savefig(boxplots_dir / 'boxplot_key_au_distribution.png', dpi=150)
    plt.close()

def create_heatmaps(patient_data, results_df, output_dir):
    heatmaps_dir = Path(output_dir) / 'heatmaps'
    heatmaps_dir.mkdir(exist_ok=True)
    
    # 1. 三患者综合对比热图
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(results_df.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, ax=ax, cbar_kws={'label': 'AU Activation'})
    ax.set_title('男性患者3M - 积极情绪AU激活热图', fontsize=14)
    ax.set_xlabel('患者')
    ax.set_ylabel('Action Units')
    plt.tight_layout()
    plt.savefig(heatmaps_dir / 'heatmap_all_patients_comparison.png', dpi=150)
    plt.close()
    
    # 2. 患者间差异热图
    diff_matrix = pd.DataFrame(index=au_columns, columns=['M1-M2', 'M1-M3', 'M2-M3'])
    for au in au_columns:
        diff_matrix.loc[au, 'M1-M2'] = results_df.loc['M1', au] - results_df.loc['M2', au]
        diff_matrix.loc[au, 'M1-M3'] = results_df.loc['M1', au] - results_df.loc['M3', au]
        diff_matrix.loc[au, 'M2-M3'] = results_df.loc['M2', au] - results_df.loc['M3', au]
    
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(diff_matrix.astype(float), annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, cbar_kws={'label': '差异值'})
    ax.set_title('男性患者3M - 积极情绪患者间AU差异热图', fontsize=14)
    plt.tight_layout()
    plt.savefig(heatmaps_dir / 'heatmap_patient_differences.png', dpi=150)
    plt.close()
    
    # 3. 各患者单独热图
    for pid in ['M1', 'M2', 'M3']:
        fig, ax = plt.subplots(figsize=(14, 2))
        patient_df = pd.DataFrame(results_df.loc[pid]).T
        patient_df.index = [pid]
        sns.heatmap(patient_df, annot=True, fmt='.2f', cmap='RdYlBu_r',
                    center=0, ax=ax, cbar_kws={'label': '激活值'})
        ax.set_title(f'男性患者{pid} - 积极情绪AU激活', fontsize=12)
        plt.tight_layout()
        plt.savefig(heatmaps_dir / f'heatmap_{pid}.png', dpi=150)
        plt.close()

def create_radar(patient_data, results_df, output_dir):
    radar_dir = Path(output_dir) / 'radar'
    radar_dir.mkdir(exist_ok=True)
    
    key_aus = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU17_r']
    angles = np.linspace(0, 2*np.pi, len(key_aus), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, pid in enumerate(['M1', 'M2', 'M3']):
        values = [results_df.loc[pid, au] for au in key_aus]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=pid, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([au.replace('_r', '') for au in key_aus])
    ax.set_ylim(0, results_df[key_aus].max().max() * 1.2)
    ax.set_title('男性患者3M - 积极情绪AU轮廓雷达图', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(radar_dir / 'radar_patient_profile.png', dpi=150)
    plt.close()

def create_time_series(patient_data, output_dir):
    ts_dir = Path(output_dir) / 'time_series'
    ts_dir.mkdir(exist_ok=True)
    
    key_aus = ['AU04_r', 'AU06_r', 'AU12_r']
    
    for au in key_aus:
        fig, ax = plt.subplots(figsize=(12, 5))
        for pid in ['M1', 'M2', 'M3']:
            data = patient_data[pid][au].values
            target_len = min(1000, len(data))
            indices = np.linspace(0, len(data)-1, target_len, dtype=int)
            data_downsampled = data[indices]
            time_sec = np.arange(len(data_downsampled)) * 0.033
            ax.plot(time_sec, data_downsampled, label=pid, alpha=0.7)
        
        ax.set_xlabel('时间 (秒)', fontsize=12)
        ax.set_ylabel(f'{au.replace("_r", "")} 激活强度', fontsize=12)
        ax.set_title(f'男性患者3M - 积极情绪{au.replace("_r", "")}时间序列对比', fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.savefig(ts_dir / f'timeseries_{au}.png', dpi=150)
        plt.close()

def save_raw_data(patient_data, output_dir):
    raw_dir = Path(output_dir) / 'raw_data'
    raw_dir.mkdir(exist_ok=True)
    
    for pid, df in patient_data.items():
        key_cols = ['frame', 'timestamp', 'confidence'] + au_columns
        df[key_cols].to_csv(raw_dir / f'{pid}.csv', index=False, encoding='utf-8-sig')

def create_statistics(patient_data, results_df, output_dir):
    stats_dir = Path(output_dir) / 'statistics'
    stats_dir.mkdir(exist_ok=True)
    
    # 计算ANOVA F值
    f_values = []
    for au in au_columns:
        m1_data = patient_data['M1'][au].values
        m2_data = patient_data['M2'][au].values
        m3_data = patient_data['M3'][au].values
        
        f_stat, p_val = stats.f_oneway(m1_data, m2_data, m3_data)
        f_values.append({'AU': au, 'F_value': f_stat, 'p_value': p_val})
    
    f_df = pd.DataFrame(f_values)
    f_df = f_df.sort_values('F_value', ascending=False)
    f_df.to_csv(stats_dir / 'au_patient_statistics.csv', index=False, encoding='utf-8-sig')
    
    # 绘制F值图
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'gray' 
              for p in f_df['p_value']]
    bars = ax.bar(range(len(f_df)), f_df['F_value'], color=colors)
    ax.set_xticks(range(len(f_df)))
    ax.set_xticklabels([au.replace('_r', '') for au in f_df['AU']], rotation=45)
    ax.set_ylabel('F值', fontsize=12)
    ax.set_title('男性患者3M - 积极情绪ANOVA F值 (红色:p<0.001, 橙色:p<0.01)', fontsize=14)
    plt.tight_layout()
    plt.savefig(stats_dir / 'statistics_anova_f_values.png', dpi=150)
    plt.close()

def generate_report(patient_data, results_df, output_dir):
    output_path = Path(output_dir) / 'analysis_report.txt'
    
    f_values = []
    for au in au_columns:
        m1_data = patient_data['M1'][au].values
        m2_data = patient_data['M2'][au].values
        m3_data = patient_data['M3'][au].values
        f_stat, p_val = stats.f_oneway(m1_data, m2_data, m3_data)
        f_values.append((au, f_stat, p_val))
    
    f_values.sort(key=lambda x: x[1], reverse=True)
    
    report = f"""================================================================================
男性患者3M积极情绪AU特征分析报告
日期: {pd.Timestamp.now().strftime('%Y-%m-%d')}
================================================================================

1. 数据概况
-----------
分析对象: 男性抑郁症患者 (M1, M2, M3)
情绪类型: 积极情绪
数据来源: OpenFace 2.0 AU强度值

样本量:
  • M1: {len(patient_data['M1'])} 帧
  • M2: {len(patient_data['M2'])} 帧
  • M3: {len(patient_data['M3'])} 帧

2. 核心发现
-----------
最具个体差异的AU (按F值排序):
"""
    
    for au, f_val, p_val in f_values[:10]:
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        report += f"  {au:10s}: F={f_val:8.1f}, p={p_val:.2e} {sig}\n"
    
    report += f"""
3. 各患者AU均值
---------------
"""
    for pid in ['M1', 'M2', 'M3']:
        report += f"\n{pid}:\n"
        for au in ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU17_r']:
            report += f"  {au}: {results_df.loc[pid, au]:.3f}\n"
    
    report += """
4. 结论
-------
男性患者在积极情绪下表现出显著的个体差异。
M3患者可能表现出习惯性皱眉特征（AU04高激活）。

================================================================================
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"报告已保存: {output_path}")

def main():
    output_dir = Path('/root/.openclaw/workspace/老年失智人群预警模式科研项目/analysis_results/2026-02-18_男患者3M_积极情绪对比')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("男性患者3M积极情绪对比分析")
    print("="*70)
    
    patient_data, results_df = analyze_patient_positive()
    
    print("\n生成条形图...")
    create_barplots(patient_data, results_df, output_dir)
    
    print("生成箱线图...")
    create_boxplots(patient_data, output_dir)
    
    print("生成热图...")
    create_heatmaps(patient_data, results_df, output_dir)
    
    print("生成雷达图...")
    create_radar(patient_data, results_df, output_dir)
    
    print("生成时间序列图...")
    create_time_series(patient_data, output_dir)
    
    print("保存原始数据...")
    save_raw_data(patient_data, output_dir)
    
    print("生成统计分析...")
    create_statistics(patient_data, results_df, output_dir)
    
    print("生成分析报告...")
    generate_report(patient_data, results_df, output_dir)
    
    print(f"\n{'='*70}")
    print(f"分析完成！结果保存在: {output_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
