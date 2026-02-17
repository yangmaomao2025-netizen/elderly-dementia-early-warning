#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
男性患者跨情绪分析（M1/M2/M3）
分析每个男性患者在悲伤/中性/积极三种情绪下的AU模式差异
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 文件映射 - 男性患者
file_mapping = {
    'M1': {
        '悲伤': '/root/.openclaw/media/inbound/file_27---2b92310a-c235-4003-a321-a0ec90aae5ba.csv',
        '中性': '/root/.openclaw/media/inbound/file_30---022dee2d-6990-4df3-aa41-a71b974bb24d.csv',
        '积极': '/root/.openclaw/media/inbound/file_24---925f9a2e-ba59-4283-829c-75d596785181.csv'
    },
    'M2': {
        '悲伤': '/root/.openclaw/media/inbound/file_28---a22ed529-d93d-4a5d-8bc6-2415bf19cbfe.csv',
        '中性': '/root/.openclaw/media/inbound/file_31---b6f4948f-3181-49b5-8ea9-02070668994c.csv',
        '积极': '/root/.openclaw/media/inbound/file_25---f701a00a-5efc-44e6-8514-4510879be7a9.csv'
    },
    'M3': {
        '悲伤': '/root/.openclaw/media/inbound/file_29---ee23776c-8da6-44c4-aba3-8a74eca4f174.csv',
        '中性': '/root/.openclaw/media/inbound/file_32---5ceabcaa-84d6-4901-83aa-79b841590151.csv',
        '积极': '/root/.openclaw/media/inbound/file_26---2b859f5a-08e2-4713-b654-c56162c1085d.csv'
    }
}

# AU列表
au_columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

au_names_short = {
    'AU01_r': 'AU01', 'AU02_r': 'AU02', 'AU04_r': 'AU04', 'AU05_r': 'AU05',
    'AU06_r': 'AU06', 'AU07_r': 'AU07', 'AU09_r': 'AU09', 'AU10_r': 'AU10',
    'AU12_r': 'AU12', 'AU14_r': 'AU14', 'AU15_r': 'AU15', 'AU17_r': 'AU17',
    'AU20_r': 'AU20', 'AU23_r': 'AU23', 'AU25_r': 'AU25', 'AU26_r': 'AU26',
    'AU45_r': 'AU45'
}

def load_data(filepath):
    """加载并清理数据"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def create_output_dirs(base_dir):
    """创建输出目录结构"""
    dirs = ['heatmaps', 'barplots', 'boxplots', 'radar', 'time_series', 'statistics', 'raw_data']
    for d in dirs:
        (base_dir / d).mkdir(parents=True, exist_ok=True)
    return {d: base_dir / d for d in dirs}

def analyze_patient_cross_emotion(patient_id, data_dict, output_dirs):
    """分析单个患者的跨情绪差异"""
    print(f"\n{'='*60}")
    print(f"分析 {patient_id} 跨情绪差异")
    print(f"{'='*60}")
    
    # 加载数据
    emotion_data = {}
    emotion_stats = {}
    
    for emotion, filepath in data_dict.items():
        df = load_data(filepath)
        emotion_data[emotion] = df
        
        # 计算统计信息
        stats = df[au_columns].mean().to_dict()
        emotion_stats[emotion] = stats
        
        print(f"\n{emotion} 情绪: {len(df)} 帧")
        for au in au_columns[:5]:
            print(f"  {au}: {stats[au]:.3f}")
    
    # 创建对比热图
    fig, ax = plt.subplots(figsize=(16, 4))
    
    comparison_data = []
    for emotion in ['悲伤', '中性', '积极']:
        if emotion in emotion_stats:
            comparison_data.append([emotion_stats[emotion][au] for au in au_columns])
        else:
            comparison_data.append([0] * len(au_columns))
    
    comparison_df = pd.DataFrame(comparison_data, 
                                  index=['悲伤', '中性', '积极'],
                                  columns=[au_names_short[au] for au in au_columns])
    
    sns.heatmap(comparison_df, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, ax=ax, cbar_kws={'label': 'AU Activation'})
    ax.set_title(f'{patient_id} 跨情绪AU激活对比', fontsize=14, pad=20)
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Emotion', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dirs['heatmaps'] / f'{patient_id}_cross_emotion_heatmap.png', dpi=150)
    plt.close()
    
    # 计算情绪间差异
    print(f"\n{patient_id} 情绪间AU差异:")
    if '悲伤' in emotion_stats and '中性' in emotion_stats:
        diff_sn = {au: emotion_stats['悲伤'][au] - emotion_stats['中性'][au] for au in au_columns}
        top_diff = sorted(diff_sn.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f"  悲伤 vs 中性 最大差异:")
        for au, val in top_diff:
            print(f"    {au}: {val:+.3f}")
    
    if '积极' in emotion_stats and '中性' in emotion_stats:
        diff_pn = {au: emotion_stats['积极'][au] - emotion_stats['中性'][au] for au in au_columns}
        top_diff = sorted(diff_pn.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f"  积极 vs 中性 最大差异:")
        for au, val in top_diff:
            print(f"    {au}: {val:+.3f}")
    
    return emotion_data, emotion_stats

def compare_patients_across_emotions(all_stats, output_dirs):
    """比较不同患者间的差异"""
    print(f"\n{'='*60}")
    print("M1 vs M2 vs M3 患者间对比")
    print(f"{'='*60}")
    
    # 为每种情绪创建对比
    for emotion in ['悲伤', '中性', '积极']:
        fig, ax = plt.subplots(figsize=(16, 5))
        
        comparison_data = []
        patient_ids = []
        
        for patient_id in ['M1', 'M2', 'M3']:
            if patient_id in all_stats and emotion in all_stats[patient_id]:
                stats = all_stats[patient_id][emotion]
                comparison_data.append([stats[au] for au in au_columns])
                patient_ids.append(patient_id)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data,
                                          index=patient_ids,
                                          columns=[au_names_short[au] for au in au_columns])
            
            sns.heatmap(comparison_df, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       center=0, ax=ax, cbar_kws={'label': 'AU Activation'})
            ax.set_title(f'{emotion}情绪 - M1/M2/M3患者AU激活对比', fontsize=14, pad=20)
            plt.tight_layout()
            plt.savefig(output_dirs['heatmaps'] / f'patients_comparison_{emotion}.png', dpi=150)
            plt.close()
            
            # 计算患者间差异
            print(f"\n{emotion}情绪患者间差异 (Top 5 AUs):")
            for i, au in enumerate(au_columns):
                values = [comparison_df.iloc[j, i] for j in range(len(patient_ids))]
                max_diff = max(values) - min(values)
                if max_diff > 0.3:  # 只显示显著差异
                    print(f"  {au}: {dict(zip(patient_ids, [f'{v:.3f}' for v in values]))} (差:{max_diff:.3f})")

def generate_correlation_heatmaps(all_data, output_dirs):
    """生成17×17 AU相关矩阵热图"""
    print(f"\n{'='*60}")
    print("生成AU相关矩阵热图")
    print(f"{'='*60}")
    
    for patient_id, emotions in all_data.items():
        print(f"\n{patient_id}:")
        for emotion, df in emotions.items():
            corr_matrix = df[au_columns].corr()
            
            # 处理NaN
            nan_count = corr_matrix.isna().sum().sum()
            if nan_count > 0:
                corr_matrix = corr_matrix.fillna(0)
            
            fig, ax = plt.subplots(figsize=(14, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                       cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                       square=True, linewidths=0.5, ax=ax)
            
            frame_count = len(df)
            ax.set_title(f'{patient_id} - {emotion}情绪 ({frame_count} frames)', fontsize=14, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            output_path = output_dirs['heatmaps'] / f'{patient_id}_{emotion}_correlation_matrix.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  {emotion}: {output_path.name}")

def generate_summary_report(all_stats, output_dirs):
    """生成汇总统计报告"""
    report_lines = ["# 男性患者跨情绪分析报告\n\n"]
    
    # 汇总每个患者的跨情绪模式
    for patient_id in ['M1', 'M2', 'M3']:
        if patient_id not in all_stats:
            continue
            
        report_lines.append(f"## {patient_id}\n\n")
        
        for emotion in ['悲伤', '中性', '积极']:
            if emotion not in all_stats[patient_id]:
                continue
            report_lines.append(f"### {emotion}情绪\n\n")
            stats = all_stats[patient_id][emotion]
            sorted_aus = sorted(stats.items(), key=lambda x: x[1], reverse=True)
            report_lines.append("| AU | 激活值 | 排名 |\n")
            report_lines.append("|-----|--------|------|\n")
            for rank, (au, val) in enumerate(sorted_aus[:10], 1):
                report_lines.append(f"| {au} | {val:.3f} | {rank} |\n")
            report_lines.append("\n")
    
    # 保存报告
    with open(output_dirs['statistics'] / 'cross_emotion_summary.md', 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print("\n汇总报告已保存")

def main():
    # 设置输出目录
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d')
    base_dir = Path(f'/root/.openclaw/workspace/analysis_results/{timestamp}_男性患者跨情绪分析')
    output_dirs = create_output_dirs(base_dir)
    
    print("="*70)
    print("男性患者跨情绪分析（M1/M2/M3）")
    print("="*70)
    
    # 存储所有统计信息和原始数据
    all_stats = {}
    all_data = {}
    
    # 分析每个患者
    for patient_id, data_dict in file_mapping.items():
        emotion_data, emotion_stats = analyze_patient_cross_emotion(
            patient_id, data_dict, output_dirs
        )
        all_stats[patient_id] = emotion_stats
        all_data[patient_id] = emotion_data
    
    # 患者间对比
    compare_patients_across_emotions(all_stats, output_dirs)
    
    # 生成相关矩阵热图
    generate_correlation_heatmaps(all_data, output_dirs)
    
    # 生成汇总报告
    generate_summary_report(all_stats, output_dirs)
    
    print(f"\n{'='*70}")
    print(f"分析完成！结果保存在: {base_dir}")
    print(f"{'='*70}")
    
    # 列出生成的文件
    for dir_name, dir_path in output_dirs.items():
        files = list(dir_path.glob('*'))
        if files:
            print(f"\n{dir_name}/: {len(files)} 个文件")

if __name__ == '__main__':
    main()
