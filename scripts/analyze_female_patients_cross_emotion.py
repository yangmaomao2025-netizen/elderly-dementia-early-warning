#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
女性患者跨情绪分析（F1/F2/F3）
分析每个女性患者在悲伤/中性/积极三种情绪下的AU模式差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 文件映射
file_mapping = {
    'F1': {
        '悲伤': '/root/.openclaw/media/inbound/file_39---a156b700-0dc5-4a7d-98a7-842a4d9ee1da.csv',
        '中性': '/root/.openclaw/media/inbound/file_33---0952895f-e350-4674-8b75-1c310d208392.csv',
        '积极': '/root/.openclaw/media/inbound/file_36---182b42a2-87e5-41cf-af26-d505d000e246.csv'
    },
    'F2': {
        '悲伤': '/root/.openclaw/media/inbound/file_40---571ec375-dfd5-403f-b0da-0922d9ff79e6.csv',
        '中性': '/root/.openclaw/media/inbound/file_34---02e0b344-096c-4493-99ba-38f427cc085a.csv',
        '积极': '/root/.openclaw/media/inbound/file_37---c15a0c41-ee72-464a-b765-40030db85592.csv'
    },
    'F3': {
        '悲伤': '/root/.openclaw/media/inbound/file_41---90b39ad4-e124-4e19-84ad-d2beb4ddcc4e.csv',
        '中性': '/root/.openclaw/media/inbound/file_35---1d02ab72-301b-4468-8579-ed6bf3c7152a.csv',
        '积极': '/root/.openclaw/media/inbound/file_38---0e31b32a-8a78-4f18-9127-33ab67cbf863.csv'
    }
}

# AU列表
au_columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

au_names = {
    'AU01_r': 'AU01\n(Inner Brow Raise)',
    'AU02_r': 'AU02\n(Outer Brow Raise)',
    'AU04_r': 'AU04\n(Brow Lower)',
    'AU05_r': 'AU05\n(Upper Lid Raise)',
    'AU06_r': 'AU06\n(Cheek Raise)',
    'AU07_r': 'AU07\n(Lid Tighten)',
    'AU09_r': 'AU09\n(Nose Wrinkle)',
    'AU10_r': 'AU10\n(Upper Lip Raise)',
    'AU12_r': 'AU12\n(Lip Corner Pull)',
    'AU14_r': 'AU14\n(Dimpler)',
    'AU15_r': 'AU15\n(Lip Corner Depress)',
    'AU17_r': 'AU17\n(Chin Raise)',
    'AU20_r': 'AU20\n(Lip Stretch)',
    'AU23_r': 'AU23\n(Lip Tighten)',
    'AU25_r': 'AU25\n(Lips Part)',
    'AU26_r': 'AU26\n(Jaw Drop)',
    'AU45_r': 'AU45\n(Blink)'
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
    fig, ax = plt.subplots(figsize=(14, 4))
    
    comparison_data = []
    for emotion in ['悲伤', '中性', '积极']:
        if emotion in emotion_stats:
            comparison_data.append([emotion_stats[emotion][au] for au in au_columns])
        else:
            comparison_data.append([0] * len(au_columns))
    
    comparison_df = pd.DataFrame(comparison_data, 
                                  index=['悲伤', '中性', '积极'],
                                  columns=[au_names[au] for au in au_columns])
    
    sns.heatmap(comparison_df, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, ax=ax, cbar_kws={'label': 'AU Activation'})
    ax.set_title(f'{patient_id} 跨情绪AU激活对比', fontsize=14, pad=20)
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Emotion', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dirs['heatmaps'] / f'{patient_id}_cross_emotion_heatmap.png', dpi=150)
    plt.close()
    
    # 计算情绪间差异
    print(f"\n{patient_id} 情绪间AU差异 (悲伤 vs 中性 vs 积极):")
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
    print("F1 vs F2 vs F3 患者间对比")
    print(f"{'='*60}")
    
    # 为每种情绪创建对比
    for emotion in ['悲伤', '中性', '积极']:
        fig, ax = plt.subplots(figsize=(16, 5))
        
        comparison_data = []
        patient_ids = []
        
        for patient_id in ['F1', 'F2', 'F3']:
            if patient_id in all_stats and emotion in all_stats[patient_id]:
                stats = all_stats[patient_id][emotion]
                comparison_data.append([stats[au] for au in au_columns])
                patient_ids.append(patient_id)
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data,
                                          index=patient_ids,
                                          columns=[au_names[au] for au in au_columns])
            
            sns.heatmap(comparison_df, annot=True, fmt='.2f', cmap='RdYlBu_r',
                       center=0, ax=ax, cbar_kws={'label': 'AU Activation'})
            ax.set_title(f'{emotion}情绪 - F1/F2/F3患者AU激活对比', fontsize=14, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dirs['heatmaps'] / f'patients_comparison_{emotion}.png', dpi=150)
            plt.close()
            
            # 计算患者间差异
            print(f"\n{emotion}情绪患者间差异:")
            for i, au in enumerate(au_columns[:5]):
                values = [comparison_df.iloc[j, i] for j in range(len(patient_ids))]
                print(f"  {au}: {dict(zip(patient_ids, [f'{v:.3f}' for v in values]))}")

def generate_summary_report(all_stats, output_dirs):
    """生成汇总统计报告"""
    report_lines = ["# 女性患者跨情绪分析报告\n",
                    "## 数据说明\n",
                    "- F1-中性数据不完整（仅294帧/10秒），结果仅供参考\n",
                    "\n## 各患者AU激活均值\n"]
    
    for patient_id in ['F1', 'F2', 'F3']:
        report_lines.append(f"\n### {patient_id}\n")
        if patient_id in all_stats:
            for emotion in ['悲伤', '中性', '积极']:
                if emotion in all_stats[patient_id]:
                    report_lines.append(f"\n{emotion}情绪:\n")
                    stats = all_stats[patient_id][emotion]
                    sorted_aus = sorted(stats.items(), key=lambda x: x[1], reverse=True)
                    for au, val in sorted_aus[:10]:
                        report_lines.append(f"- {au}: {val:.3f}\n")
    
    # 保存报告
    with open(output_dirs['statistics'] / 'cross_emotion_summary.md', 'w', encoding='utf-8') as f:
        f.writelines(report_lines)
    
    print("\n汇总报告已保存")

def main():
    # 设置输出目录
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d')
    base_dir = Path(f'/root/.openclaw/workspace/analysis_results/{timestamp}_女性患者跨情绪分析')
    output_dirs = create_output_dirs(base_dir)
    
    print("="*70)
    print("女性患者跨情绪分析（F1/F2/F3）")
    print("="*70)
    
    # 存储所有统计信息
    all_stats = {}
    
    # 分析每个患者
    for patient_id, data_dict in file_mapping.items():
        emotion_data, emotion_stats = analyze_patient_cross_emotion(
            patient_id, data_dict, output_dirs
        )
        all_stats[patient_id] = emotion_stats
    
    # 患者间对比
    compare_patients_across_emotions(all_stats, output_dirs)
    
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
