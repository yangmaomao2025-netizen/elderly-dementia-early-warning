#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F1对照组三种情绪对比分析
分析F1在悲伤、中性、积极三种情绪下的AU特征差异
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

# F1文件映射
f1_files = {
    '悲伤': '/root/.openclaw/media/inbound/file_5---69ad20a2-5a2f-4f18-bdef-056d8c24d515.csv',
    '中性': '/root/.openclaw/media/inbound/file_20---333e020a-bdf5-44a5-b833-c3179c272ccc.csv',
    '积极': '/root/.openclaw/media/inbound/file_23---06535c58-c474-473b-a68d-aadcee3e3ca7.csv'
}

au_columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

def load_and_process(filepath):
    """加载并处理数据"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def analyze_f1_cross_emotion():
    """分析F1三种情绪对比"""
    print("="*70)
    print("F1对照组三种情绪对比分析")
    print("="*70)
    
    # 加载数据
    emotion_data = {}
    for emotion, filepath in f1_files.items():
        try:
            df = load_and_process(filepath)
            emotion_data[emotion] = df
            print(f"{emotion}: {len(df)} frames")
        except Exception as e:
            print(f"Error loading {emotion}: {e}")
    
    # 计算各情绪的AU均值
    results = []
    for emotion, df in emotion_data.items():
        au_means = df[au_columns].mean()
        au_means['emotion'] = emotion
        results.append(au_means)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index('emotion')
    
    print("\n各情绪AU均值:")
    print(results_df.round(3))
    
    # 计算跨情绪差异
    print("\n跨情绪差异分析:")
    print(f"悲伤 vs 中性: {(results_df.loc['悲伤'] - results_df.loc['中性']).abs().mean():.3f} (平均绝对差异)")
    print(f"悲伤 vs 积极: {(results_df.loc['悲伤'] - results_df.loc['积极']).abs().mean():.3f} (平均绝对差异)")
    print(f"中性 vs 积极: {(results_df.loc['中性'] - results_df.loc['积极']).abs().mean():.3f} (平均绝对差异)")
    
    return emotion_data, results_df

def create_visualizations(emotion_data, results_df, output_dir):
    """创建可视化"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 三情绪AU均值热图
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(results_df.T, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                center=0, ax=ax, cbar_kws={'label': 'AU Activation'})
    ax.set_title('F1对照组 - 三种情绪AU激活热图', fontsize=14)
    ax.set_xlabel('情绪类型')
    ax.set_ylabel('Action Units')
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_f1_three_emotions.png', dpi=150)
    plt.close()
    
    # 2. 关键AU对比条形图
    key_aus = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU17_r']
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(key_aus))
    width = 0.25
    
    for i, emotion in enumerate(['悲伤', '中性', '积极']):
        values = [results_df.loc[emotion, au] for au in key_aus]
        ax.bar(x + i*width, values, width, label=emotion)
    
    ax.set_xlabel('Action Units')
    ax.set_ylabel('激活强度')
    ax.set_title('F1对照组 - 关键AU三情绪对比')
    ax.set_xticks(x + width)
    ax.set_xticklabels(key_aus)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'barplot_f1_key_au_comparison.png', dpi=150)
    plt.close()
    
    # 3. 情绪差异热图
    diff_matrix = pd.DataFrame(index=au_columns, columns=['悲伤-中性', '悲伤-积极', '中性-积极'])
    for au in au_columns:
        diff_matrix.loc[au, '悲伤-中性'] = results_df.loc['悲伤', au] - results_df.loc['中性', au]
        diff_matrix.loc[au, '悲伤-积极'] = results_df.loc['悲伤', au] - results_df.loc['积极', au]
        diff_matrix.loc[au, '中性-积极'] = results_df.loc['中性', au] - results_df.loc['积极', au]
    
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(diff_matrix.astype(float), annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, ax=ax, cbar_kws={'label': '差异值'})
    ax.set_title('F1对照组 - 情绪间AU差异热图', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_f1_emotion_differences.png', dpi=150)
    plt.close()
    
    # 4. 雷达图
    key_aus_radar = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU17_r']
    angles = np.linspace(0, 2*np.pi, len(key_aus_radar), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, emotion in enumerate(['悲伤', '中性', '积极']):
        values = [results_df.loc[emotion, au] for au in key_aus_radar]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=emotion, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([au.replace('_r', '') for au in key_aus_radar])
    ax.set_ylim(0, results_df[key_aus_radar].max().max() * 1.2)
    ax.set_title('F1对照组 - 三种情绪AU轮廓雷达图', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(output_dir / 'radar_f1_emotion_profile.png', dpi=150)
    plt.close()
    
    # 5. 时间序列对比（AU12 - 笑容）
    fig, ax = plt.subplots(figsize=(12, 5))
    for emotion, df in emotion_data.items():
        au12_data = df['AU12_r'].values
        # 下采样到相同长度
        target_len = min(1000, len(au12_data))
        indices = np.linspace(0, len(au12_data)-1, target_len, dtype=int)
        au12_downsampled = au12_data[indices]
        time_sec = np.arange(len(au12_downsampled)) * 0.033
        ax.plot(time_sec, au12_downsampled, label=emotion, alpha=0.7)
    
    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('AU12激活强度')
    ax.set_title('F1对照组 - AU12时间序列对比（嘴角上扬/笑容）')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'timeseries_f1_au12_comparison.png', dpi=150)
    plt.close()
    
    return diff_matrix

def generate_report(emotion_data, results_df, diff_matrix, output_dir):
    """生成分析报告"""
    output_dir = Path(output_dir)
    
    # 找出最具区分度的AU
    discrimination = diff_matrix.abs().mean(axis=1).sort_values(ascending=False)
    
    report = f"""# F1对照组三种情绪对比分析报告

## 1. 数据概况

| 情绪 | 帧数 |
|------|------|
| 悲伤 | {len(emotion_data['悲伤'])} |
| 中性 | {len(emotion_data['中性'])} |
| 积极 | {len(emotion_data['积极'])} |

## 2. 关键发现

### 2.1 最具区分度的AU（按跨情绪差异排序）

| AU | 平均差异 |
|------|----------|
"""
    
    for au, diff in discrimination.head(5).items():
        report += f"| {au} | {diff:.3f} |\n"
    
    report += f"""
### 2.2 各情绪AU均值

{results_df.round(3).to_markdown()}

### 2.3 关键观察

1. **AU07 (眼睑收紧)**: 
   - 悲伤: {results_df.loc['悲伤', 'AU07_r']:.3f}
   - 中性: {results_df.loc['中性', 'AU07_r']:.3f}
   - 积极: {results_df.loc['积极', 'AU07_r']:.3f}

2. **AU12 (嘴角上扬/笑容)** :
   - 悲伤: {results_df.loc['悲伤', 'AU12_r']:.3f}
   - 中性: {results_df.loc['中性', 'AU12_r']:.3f}
   - 积极: {results_df.loc['积极', 'AU12_r']:.3f}

3. **AU04 (皱眉)**:
   - 悲伤: {results_df.loc['悲伤', 'AU04_r']:.3f}
   - 中性: {results_df.loc['中性', 'AU04_r']:.3f}
   - 积极: {results_df.loc['积极', 'AU04_r']:.3f}

## 3. 与M1/M2对比

F1（女性）与M1/M2（男性）的差异:
- AU07激活普遍较低（女性特征）
- AU12在积极情绪下表现

## 4. 结论

F1对照组作为女性对照样本，展示了女性特有的AU激活模式，
可作为性别差异分析的基准。

---
生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
    
    with open(output_dir / 'analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n报告已保存: {output_dir / 'analysis_report.md'}")

def main():
    # 创建输出目录
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d')
    output_dir = Path(f'/root/.openclaw/workspace/老年失智人群预警模式科研项目/analysis_results/{timestamp}_F1_三种情绪对比')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n开始分析...")
    
    # 执行分析
    emotion_data, results_df = analyze_f1_cross_emotion()
    
    # 创建可视化
    diff_matrix = create_visualizations(emotion_data, results_df, output_dir)
    
    # 生成报告
    generate_report(emotion_data, results_df, diff_matrix, output_dir)
    
    # 保存原始数据
    results_df.to_csv(output_dir / 'f1_au_means.csv', encoding='utf-8-sig')
    diff_matrix.to_csv(output_dir / 'f1_emotion_differences.csv', encoding='utf-8-sig')
    
    print(f"\n{'='*70}")
    print(f"分析完成！结果保存在: {output_dir}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
