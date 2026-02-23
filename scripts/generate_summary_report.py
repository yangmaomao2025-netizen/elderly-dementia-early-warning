#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合可视化汇总报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

print("="*60)
print("生成综合可视化汇总报告")
print("="*60)

output_dir = "2025-02-23_综合可视化汇总报告"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}/figures", exist_ok=True)

# 读取统计数据
stats_df = pd.read_csv('2025-02-23_患者组vs正常组_统计检验/statistics/group_comparison_statistics.csv')

# 1. 三情绪差异对比总览
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

emotions = ['悲伤', '积极', '中性']
colors = ['#3498db', '#e74c3c', '#95a5a6']

for ax, emotion, color in zip(axes, emotions, colors):
    emotion_data = stats_df[stats_df['emotion'] == emotion].copy()
    emotion_data['neg_log_p'] = -np.log10(emotion_data['p_value'].clip(lower=1e-300))
    
    # 显著性着色
    point_colors = []
    for _, row in emotion_data.iterrows():
        if row['p_value'] < 0.05:
            point_colors.append('#e74c3c' if row['cohens_d'] > 0 else '#3498db')
        else:
            point_colors.append('#95a5a6')
    
    ax.scatter(emotion_data['cohens_d'], emotion_data['neg_log_p'], 
              c=point_colors, alpha=0.7, s=100, edgecolors='white', linewidth=0.5)
    
    ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel("Cohen's d", fontsize=11)
    ax.set_ylabel('-log10(p)', fontsize=11)
    ax.set_title(f'{emotion}情绪', fontsize=13, fontweight='bold')
    
    # 标注显著差异
    sig_data = emotion_data[emotion_data['p_value'] < 0.05]
    for _, row in sig_data.iterrows():
        ax.annotate(row['AU'], (row['cohens_d'], row['neg_log_p']), 
                   fontsize=8, ha='center', fontweight='bold')

plt.suptitle('患者组 vs 正常对照组 - 三情绪差异对比', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/figures/01_三情绪火山图总览.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 三情绪火山图总览")

# 2. 显著差异AU汇总
sig_df = stats_df[stats_df['p_value'] < 0.05].copy()

if len(sig_df) > 0:
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(sig_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sig_df['patient_mean'], width, label='患者组', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, sig_df['normal_mean'], width, label='正常对照组', color='#3498db', alpha=0.8)
    
    # 添加标签
    labels = [f"{row['emotion']}\n{row['AU']}" for _, row in sig_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title('显著差异AU汇总 (p < 0.05)', fontsize=14, fontweight='bold')
    ax.legend()
    
    # 添加效应量标注
    for i, (_, row) in enumerate(sig_df.iterrows()):
        ax.text(i, max(row['patient_mean'], row['normal_mean']) * 1.1, 
               f"d={row['cohens_d']:.2f}", ha='center', fontsize=8, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figures/02_显著差异AU汇总.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 显著差异AU汇总")

# 3. 效应量分布
fig, ax = plt.subplots(figsize=(10, 6))

colors_map = {'悲伤': '#3498db', '积极': '#e74c3c', '中性': '#95a5a6'}
for emotion in emotions:
    emotion_data = stats_df[stats_df['emotion'] == emotion]
    ax.hist(emotion_data['cohens_d'], bins=15, alpha=0.6, label=emotion, color=colors_map[emotion])

ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.3)

ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('效应量分布 (患者组 - 正常对照组)', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/figures/03_效应量分布.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 效应量分布")

# 4. 各组平均AU轮廓
fig, ax = plt.subplots(figsize=(16, 6))

au_cols = stats_df['AU'].unique()
x = np.arange(len(au_cols))
width = 0.25

for i, emotion in enumerate(emotions):
    emotion_data = stats_df[stats_df['emotion'] == emotion]
    patient_means = emotion_data.groupby('AU')['patient_mean'].mean().values
    offset = (i - 1) * width
    ax.bar(x + offset, patient_means, width, label=f'{emotion}(患者)', alpha=0.8, color=colors[i])

ax.set_xlabel('Action Units', fontsize=12)
ax.set_ylabel('Mean Intensity', fontsize=12)
ax.set_title('患者组 - 各情绪AU轮廓', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(au_cols, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/figures/04_患者组AU轮廓.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 患者组AU轮廓")

# 5. 正常组AU轮廓
fig, ax = plt.subplots(figsize=(16, 6))

for i, emotion in enumerate(emotions):
    emotion_data = stats_df[stats_df['emotion'] == emotion]
    normal_means = emotion_data.groupby('AU')['normal_mean'].mean().values
    offset = (i - 1) * width
    ax.bar(x + offset, normal_means, width, label=f'{emotion}(正常)', alpha=0.8, color=colors[i])

ax.set_xlabel('Action Units', fontsize=12)
ax.set_ylabel('Mean Intensity', fontsize=12)
ax.set_title('正常对照组 - 各情绪AU轮廓', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(au_cols, rotation=45, ha='right')
ax.legend()
plt.tight_layout()
plt.savefig(f'{output_dir}/figures/05_正常组AU轮廓.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 正常组AU轮廓")

# 6. 生成综合报告
print("\n生成综合报告...")

report = f"""================================================================================
老年失智人群预警模式科研项目 - 综合可视化汇总报告
分析日期: {datetime.now().strftime('%Y-%m-%d')}
================================================================================

一、项目概述
-----------
本项目旨在通过面部动作单元(AU)分析，探索阿尔茨海默病患者与正常老年人在不同
情绪状态下的面部表达差异。

二、样本信息
-----------
患者组 (n=4): ZFL, MHD, WGL, ZJK
正常对照组 (n=4): M1, M2, F1, F2
分析指标: 17个OpenFace AU强度指标
情绪类型: 悲伤、积极、中性

三、主要发现
-----------

3.1 组间显著差异 (p < 0.05)
总显著差异AU: {len(sig_df)} / {len(stats_df)}

按情绪分类:
  悲伤情绪: {len(sig_df[sig_df['emotion']=='悲伤'])} 个AU显著差异
  积极情绪: {len(sig_df[sig_df['emotion']=='积极'])} 个AU显著差异
  中性情绪: {len(sig_df[sig_df['emotion']=='中性'])} 个AU显著差异

3.2 最显著差异的AU (按p值排序)
"""

for i, (_, row) in enumerate(sig_df.head(10).iterrows(), 1):
    direction = "患者组更高" if row['cohens_d'] > 0 else "正常组更高"
    report += f"\n  {i}. {row['emotion']}-{row['AU']}: p={row['p_value']:.2e}, d={row['cohens_d']:.2f} ({direction})"
    report += f"\n     患者组: {row['patient_mean']:.3f} ± {row['patient_std']:.3f}"
    report += f"\n     正常组: {row['normal_mean']:.3f} ± {row['normal_std']:.3f}"

report += f"""

3.3 关键发现
"""

# 患者组高表达
patient_higher = sig_df[sig_df['cohens_d'] > 0]
if len(patient_higher) > 0:
    report += "\n患者组显著高表达的AU:\n"
    for _, row in patient_higher.iterrows():
        report += f"  - {row['emotion']} {row['AU']}: d={row['cohens_d']:.2f}\n"

# 正常组高表达
normal_higher = sig_df[sig_df['cohens_d'] < 0]
if len(normal_higher) > 0:
    report += "\n正常组显著高表达的AU:\n"
    for _, row in normal_higher.iterrows():
        report += f"  - {row['emotion']} {row['AU']}: d={row['cohens_d']:.2f}\n"

report += f"""

四、AU功能解读
-------------
"""

au_functions = {
    'AU01': '眉毛内侧提升',
    'AU02': '眉毛外侧提升',
    'AU04': '眉毛下沉',
    'AU05': '上眼睑提升',
    'AU06': '脸颊提升',
    'AU07': '眼睑收紧',
    'AU09': '鼻子皱起',
    'AU10': '上唇提升',
    'AU12': '嘴角上扬',
    'AU14': '嘴角下压',
    'AU15': '嘴角下拉',
    'AU17': '下巴抬起',
    'AU20': '嘴唇拉伸',
    'AU23': '嘴唇收紧',
    'AU25': '嘴唇分开',
    'AU26': '下巴下沉',
    'AU45': '眨眼'
}

for _, row in sig_df.iterrows():
    au = row['AU'].replace('_r', '')
    func = au_functions.get(au, '')
    report += f"\n{au}: {func}"
    if row['cohens_d'] < 0:
        report += f" - 正常组显著高表达 (d={row['cohens_d']:.2f})"
    else:
        report += f" - 患者组显著高表达 (d={row['cohens_d']:.2f})"

report += f"""

五、输出图表清单
---------------
figures/
  01_三情绪火山图总览.png
  02_显著差异AU汇总.png
  03_效应量分布.png
  04_患者组AU轮廓.png
  05_正常组AU轮廓.png

六、研究意义与建议
----------------
1. 研究发现正常组在多个AU上表达强度显著高于患者组，提示阿尔茨海默病患者
   面部表情可能存在"钝化"现象。

2. 悲伤情绪下，正常组在AU20(嘴唇拉伸)、AU26(下巴下沉)、AU23(嘴唇收紧)
   上表达更强，可能反映更丰富的情绪调节能力。

3. 积极情绪下，正常组在AU17(下巴抬起)、AU15(嘴角下拉)、AU26(下巴下沉)
   上表达更强，提示患者积极情绪表达相对单一。

4. 建议后续研究：
   - 扩大样本量进行验证
   - 结合临床症状评分进行相关性分析
   - 探索AU动态变化特征
   - 开发基于AU的辅助诊断工具

================================================================================
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================
"""

with open(f'{output_dir}/综合可视化汇总报告.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n{'='*60}")
print(f"✓ 综合可视化汇总报告完成!")
print(f"输出目录: {output_dir}")
print(f"{'='*60}")
