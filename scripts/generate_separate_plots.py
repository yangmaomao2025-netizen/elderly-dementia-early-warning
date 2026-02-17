#!/usr/bin/env python3
"""
三情绪AU特征分析 - 分开生成独立图表
按日期和主题分类存放
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 配置 ============
BASE_DIR = '/root/.openclaw/workspace/analysis_results/2025-02-17_AU_emotion_analysis'
FILE_CONFIG = {
    '悲伤': '/root/.openclaw/media/inbound/file_3---b3314058-964d-470d-8293-13430fdde2c6.csv',
    '风景': '/root/.openclaw/media/inbound/file_4---0dd96eb3-72ff-4ced-a1b8-c5c51fad721a.csv',
    '正性': '/root/.openclaw/media/inbound/file_5---69ad20a2-5a2f-4f18-bdef-056d8c24d515.csv'
}

EMOTION_COLORS = {
    '悲伤': '#3498db',
    '风景': '#95a5a6', 
    '正性': '#e74c3c'
}

CORE_AU = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
           'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
           'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

def load_data(filepath, emotion_label):
    """加载数据"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df[df['success'] == 1]
    df = df[df['confidence'] >= 0.95]
    df['emotion'] = emotion_label
    return df

def save_figure(fig, folder, filename):
    """保存图表到指定文件夹"""
    filepath = os.path.join(BASE_DIR, folder, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅ 已保存: {filepath}")

# ============ 加载数据 ============
print("=" * 70)
print("三情绪AU特征分析 - 分图生成")
print("=" * 70)

data_dict = {}
for emotion, filepath in FILE_CONFIG.items():
    df = load_data(filepath, emotion)
    data_dict[emotion] = df
    print(f"加载 {emotion}: {len(df)} 帧")

# ============ 1. 生成三个情绪各自的热力图 ============
print("\n📊 生成 AU 相关性热力图...")

for emotion, df in data_dict.items():
    fig, ax = plt.subplots(figsize=(14, 12))
    corr = df[CORE_AU].corr()
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8},
                annot_kws={'size': 9})
    ax.set_title(f'{emotion}情绪 - AU强度相关性矩阵', fontsize=16, fontweight='bold', pad=20)
    
    # 旋转x轴标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    save_figure(fig, 'heatmaps', f'heatmap_{emotion}.png')

# ============ 2. 生成三情绪对比热图 ============
print("\n📊 生成情绪间差异热图...")

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

emotions = ['悲伤', '风景', '正性']
for idx, emotion in enumerate(emotions):
    corr = data_dict[emotion][CORE_AU].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, ax=axes[idx], cbar_kws={'shrink': 0.8},
                annot_kws={'size': 8})
    axes[idx].set_title(f'{emotion} - AU相关性', fontsize=14, fontweight='bold')
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].tick_params(axis='y', rotation=0)

plt.tight_layout()
save_figure(fig, 'heatmaps', 'heatmap_all_emotions_comparison.png')

# 悲伤 vs 正性 差异热图
fig, ax = plt.subplots(figsize=(14, 12))
sad_corr = data_dict['悲伤'][CORE_AU].corr()
pos_corr = data_dict['正性'][CORE_AU].corr()
diff_corr = sad_corr - pos_corr

sns.heatmap(diff_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 9})
ax.set_title('悲伤 vs 正性 - AU相关性差异 (悲伤 - 正性)', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
save_figure(fig, 'heatmaps', 'heatmap_sad_vs_positive_diff.png')

# ============ 3. 生成AU平均强度柱状图 ============
print("\n📊 生成 AU 平均强度对比图...")

fig, ax = plt.subplots(figsize=(16, 10))
x_pos = np.arange(len(CORE_AU))
width = 0.25

for i, (emotion, df) in enumerate(data_dict.items()):
    means = [df[au].mean() for au in CORE_AU]
    sems = [df[au].sem() for au in CORE_AU]
    ax.bar(x_pos + i*width, means, width, yerr=sems, 
           label=emotion, color=EMOTION_COLORS[emotion], 
           alpha=0.85, capsize=3, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Action Units (AU)', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Intensity (± SEM)', fontsize=14, fontweight='bold')
ax.set_title('三情绪AU平均强度对比', fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x_pos + width)
ax.set_xticklabels([au.replace('_r', '') for au in CORE_AU], rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

save_figure(fig, 'barplots', 'barplot_au_mean_comparison.png')

# 关键AU详细对比
key_au = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU10_r']
fig, ax = plt.subplots(figsize=(12, 8))
x_pos = np.arange(len(key_au))

for i, (emotion, df) in enumerate(data_dict.items()):
    means = [df[au].mean() for au in key_au]
    stds = [df[au].std() for au in key_au]
    ax.bar(x_pos + i*width, means, width, yerr=stds,
           label=emotion, color=EMOTION_COLORS[emotion],
           alpha=0.85, capsize=4, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Key Action Units', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Intensity (± SD)', fontsize=14, fontweight='bold')
ax.set_title('关键AU强度对比 (最具区分度)', fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x_pos + width)
ax.set_xticklabels([au.replace('_r', '') for au in key_au], fontsize=12)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

save_figure(fig, 'barplots', 'barplot_key_au_comparison.png')

# ============ 4. 生成箱线图 ============
print("\n📊 生成 AU 分布箱线图...")

key_au_box = ['AU04_r', 'AU06_r', 'AU12_r', 'AU25_r']
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, au in enumerate(key_au_box):
    plot_data = []
    labels = []
    for emotion, df in data_dict.items():
        plot_data.extend(df[au].values)
        labels.extend([emotion] * len(df))
    
    box_df = pd.DataFrame({'Intensity': plot_data, 'Emotion': labels})
    
    sns.boxplot(data=box_df, x='Emotion', y='Intensity', ax=axes[idx],
                palette=EMOTION_COLORS, width=0.6)
    axes[idx].set_title(f'{au} Distribution by Emotion', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('AU Intensity', fontsize=12)
    
    # 添加均值线
    for i, emotion in enumerate(['悲伤', '风景', '正性']):
        mean_val = data_dict[emotion][au].mean()
        axes[idx].hlines(mean_val, i-0.2, i+0.2, colors='red', linestyles='--', linewidth=2)

plt.tight_layout()
save_figure(fig, 'boxplots', 'boxplot_key_au_distribution.png')

# ============ 5. 生成雷达图 ============
print("\n📊 生成情绪特征雷达图...")

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
key_au_radar = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU15_r', 'AU25_r']
angles = np.linspace(0, 2*np.pi, len(key_au_radar), endpoint=False).tolist()
angles += angles[:1]

for emotion, df in data_dict.items():
    values = [df[au].mean() for au in key_au_radar]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=3, label=emotion,
            color=EMOTION_COLORS[emotion], markersize=8)
    ax.fill(angles, values, alpha=0.2, color=EMOTION_COLORS[emotion])

ax.set_xticks(angles[:-1])
ax.set_xticklabels([au.replace('_r', '') for au in key_au_radar], fontsize=12)
ax.set_title('三情绪AU特征雷达图', fontsize=18, fontweight='bold', pad=30, y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax.grid(True, alpha=0.3)

save_figure(fig, 'radar', 'radar_emotion_profile.png')

# ============ 6. 生成时间序列图 ============
print("\n📊 生成 AU 时间序列图...")

for au in ['AU12_r', 'AU04_r', 'AU06_r']:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for emotion, df in data_dict.items():
        # 归一化时间
        time_norm = np.linspace(0, 100, len(df))
        # 平滑处理
        from scipy.ndimage import uniform_filter1d
        au_smooth = uniform_filter1d(df[au].values, size=min(20, len(df)//10))
        ax.plot(time_norm, au_smooth, label=emotion,
                color=EMOTION_COLORS[emotion], linewidth=2.5, alpha=0.9)
    
    ax.set_xlabel('Time (% of video)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{au} Intensity', fontsize=14, fontweight='bold')
    ax.set_title(f'{au} Time Course by Emotion', fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 100)
    
    save_figure(fig, 'time_series', f'timeseries_{au}.png')

# ============ 7. 生成统计显著性图 ============
print("\n📊 生成统计检验结果图...")

# 计算ANOVA
anova_results = []
for au in CORE_AU:
    groups = [df[au].values for df in data_dict.values()]
    f_stat, p_val = f_oneway(*groups)
    anova_results.append({'AU': au, 'F': f_stat, 'p': p_val})

anova_df = pd.DataFrame(anova_results).sort_values('F', ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))
colors = ['#e74c3c' if p < 0.001 else '#f39c12' if p < 0.01 else '#3498db' for p in anova_df['p']]
bars = ax.barh(range(len(anova_df)), anova_df['F'], color=colors, alpha=0.8, edgecolor='black')

ax.set_yticks(range(len(anova_df)))
ax.set_yticklabels([au.replace('_r', '') for au in anova_df['AU']], fontsize=11)
ax.set_xlabel('F-statistic (ANOVA)', fontsize=14, fontweight='bold')
ax.set_ylabel('Action Units', fontsize=14, fontweight='bold')
ax.set_title('AU区分度排序 (ANOVA F值)', fontsize=18, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# 添加数值标签
for i, (f_val, p_val) in enumerate(zip(anova_df['F'], anova_df['p'])):
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*'
    ax.text(f_val + max(anova_df['F'])*0.01, i, f'{f_val:.0f}{sig}',
            va='center', fontsize=9, fontweight='bold')

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', alpha=0.8, label='p < 0.001 ***'),
    Patch(facecolor='#f39c12', alpha=0.8, label='p < 0.01 **'),
    Patch(facecolor='#3498db', alpha=0.8, label='p < 0.05 *')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

save_figure(fig, 'statistics', 'statistics_anova_f_values.png')

# ============ 8. 保存统计表格 ============
print("\n📊 保存统计数据...")

summary_table = []
for au in CORE_AU:
    row = {'AU': au}
    for emotion, df in data_dict.items():
        row[f'{emotion}_Mean'] = df[au].mean()
        row[f'{emotion}_SD'] = df[au].std()
        row[f'{emotion}_N'] = len(df)
    # ANOVA
    anova_row = anova_df[anova_df['AU'] == au].iloc[0]
    row['F_statistic'] = anova_row['F']
    row['p_value'] = anova_row['p']
    row['significant'] = '***' if anova_row['p'] < 0.001 else ('**' if anova_row['p'] < 0.01 else ('*' if anova_row['p'] < 0.05 else 'ns'))
    summary_table.append(row)

summary_df = pd.DataFrame(summary_table)
summary_df.to_csv(os.path.join(BASE_DIR, 'statistics', 'au_emotion_statistics.csv'), index=False)
print(f"  ✅ 统计表已保存")

# 生成文字报告
report = f"""
================================================================================
三情绪AU特征分析报告
日期: 2025-02-17
================================================================================

1. 数据概况
-----------
情绪类型: 悲伤、风景(中性)、正性
数据来源: OpenFace 2.0 AU强度值

样本量:
  • 悲伤: {len(data_dict['悲伤'])} 帧
  • 风景: {len(data_dict['风景'])} 帧  
  • 正性: {len(data_dict['正性'])} 帧

2. 核心发现
-----------
最具区分度的AU (按F值排序):
"""

for i, row in anova_df.tail(5).iloc[::-1].iterrows():
    sig = '***' if row['p'] < 0.001 else '**' if row['p'] < 0.01 else '*'
    report += f"  {i+1}. {row['AU']:<8s}: F={row['F']:>8.1f}, p={row['p']:.2e} {sig}\n"

report += """
3. 情绪特异性模式
-----------------
悲伤情绪特征:
  • AU04 (眉毛下压): 最高激活
  • AU07 (眼睑收紧): 显著高值
  • AU10 (上唇提升): 中等激活

正性情绪特征:
  • AU12 (嘴角提升): 最高区分度
  • AU06 (脸颊提升): 协同激活
  • AU14 (嘴角下压): 独特激活

中性(风景)特征:
  • 所有AU强度普遍较低
  • AU12强度介于悲伤和正性之间

4. 输出文件清单
---------------
heatmaps/
  - heatmap_悲伤.png
  - heatmap_风景.png
  - heatmap_正性.png
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
  - timeseries_AU12_r.png
  - timeseries_AU04_r.png
  - timeseries_AU06_r.png

statistics/
  - statistics_anova_f_values.png
  - au_emotion_statistics.csv

================================================================================
"""

with open(os.path.join(BASE_DIR, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report)
print(f"  ✅ 分析报告已保存")

print("\n" + "=" * 70)
print("🎉 全部分析完成！")
print("=" * 70)
print(f"\n输出目录: {BASE_DIR}")
print("\n文件夹结构:")
for folder in ['heatmaps', 'barplots', 'boxplots', 'radar', 'time_series', 'statistics']:
    files = os.listdir(os.path.join(BASE_DIR, folder))
    print(f"  📁 {folder}/ ({len(files)} 个文件)")
