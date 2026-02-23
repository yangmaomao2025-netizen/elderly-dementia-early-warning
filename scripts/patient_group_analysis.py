#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
患者组跨被试分析 - 简化版（内存优化）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

print("="*60)
print("患者组跨被试分析 (简化版)")
print("="*60)

output_dir = "2025-02-23_患者组_跨被试对比"
os.makedirs(f"{output_dir}/statistics", exist_ok=True)
os.makedirs(f"{output_dir}/heatmaps", exist_ok=True)
os.makedirs(f"{output_dir}/barplots", exist_ok=True)

# 被试文件配置
subjects = {
    'ZFL': ('ZFL_三情绪对比/raw_data/ZFL_悲伤.csv', 'ZFL_三情绪对比/raw_data/ZFL_积极.csv', 'ZFL_三情绪对比/raw_data/ZFL_中性.csv'),
    'MHD': ('../sj/MHD_悲伤.csv', '../sj/MHD_积极.csv', '../sj/MHD_中性.csv'),
    'WGL': ('../sj/WGL_悲伤.csv', '../sj/WGL_积极.csv', '../sj/WGL_中性.csv'),
    'ZJK': ('../sj/ZJK_悲伤.csv', '../sj/ZJK_积极.csv', '../sj/ZJK_中性.csv')
}

emotions = ['悲伤', '积极', '中性']

# 逐被试读取并计算统计量
print("\n逐被试计算AU均值...")
subject_stats = []

for subject, (sad_f, pos_f, neu_f) in subjects.items():
    print(f"  处理 {subject}...")
    
    for emotion, file in zip(emotions, [sad_f, pos_f, neu_f]):
        df = pd.read_csv(file, on_bad_lines='skip')
        
        # 获取AU列
        au_cols = [c for c in df.columns if c.endswith('_r')]
        
        row = {'subject': subject, 'emotion': emotion, 'n_frames': len(df)}
        for au in au_cols:
            row[au.strip()] = df[au].mean()
        subject_stats.append(row)
        
        # 删除df释放内存
        del df

stats_df = pd.DataFrame(subject_stats)
stats_df.to_csv(f'{output_dir}/statistics/subject_emotion_au_means.csv', index=False)

print(f"✓ 完成: {len(stats_df)} 条记录")

# AU标签
au_cols = [c for c in stats_df.columns if c not in ['subject', 'emotion', 'n_frames']]
au_labels = [c.replace('_r', '') for c in au_cols]

# 1. 各情绪被试对比热力图
print("\n生成热力图...")
for emotion in emotions:
    emotion_data = stats_df[stats_df['emotion']==emotion]
    
    fig, ax = plt.subplots(figsize=(16, 6))
    heatmap_data = emotion_data[au_cols].values
    
    sns.heatmap(heatmap_data, 
               xticklabels=au_labels,
               yticklabels=emotion_data['subject'].values,
               annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
               cbar_kws={'label': 'Mean Intensity'})
    ax.set_title(f'{emotion} - Subject × AU Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmaps/heatmap_subject_au_{emotion}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {emotion} 热力图")

# 2. 组平均对比热力图
fig, ax = plt.subplots(figsize=(16, 6))
avg_data = []
for emotion in emotions:
    emotion_data = stats_df[stats_df['emotion']==emotion]
    avg_row = [emotion_data[au].mean() for au in au_cols]
    avg_data.append(avg_row)

heatmap_df = pd.DataFrame(avg_data, columns=au_labels, index=emotions)
sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
           cbar_kws={'label': 'Mean Intensity'})
ax.set_title('患者组 - Average AU by Emotion', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/heatmaps/heatmap_group_emotion_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 组平均热力图")

# 3. 计算变异系数
cv_data = []
for emotion in emotions:
    emotion_stats = stats_df[stats_df['emotion']==emotion]
    for au in au_cols:
        values = emotion_stats[au].values
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        cv_data.append({'emotion': emotion, 'AU': au, 'cv': cv, 
                       'mean': np.mean(values), 'std': np.std(values)})

cv_df = pd.DataFrame(cv_data)
cv_df.to_csv(f'{output_dir}/statistics/emotion_subject_variability.csv', index=False)

# 变异系数热力图
fig, ax = plt.subplots(figsize=(16, 6))
cv_matrix = []
for emotion in emotions:
    emotion_cv = cv_df[cv_df['emotion']==emotion]['cv'].values
    cv_matrix.append(emotion_cv)

cv_heatmap_df = pd.DataFrame(cv_matrix, columns=au_labels, index=emotions)
sns.heatmap(cv_heatmap_df, annot=True, fmt='.3f', cmap='Reds', ax=ax,
           cbar_kws={'label': 'Coefficient of Variation'})
ax.set_title('患者组 - Inter-Subject Variability (CV)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/heatmaps/heatmap_inter_subject_cv.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 变异系数热力图")

# 4. 生成报告
print("\n生成分析报告...")
avg_cv = cv_df.groupby('emotion')['cv'].mean()

report = f"""================================================================================
跨被试组分析报告
组名: 患者组
被试: ZFL, MHD, WGL, ZJK
日期: {datetime.now().strftime('%Y-%m-%d')}
================================================================================

1. 数据概况
-----------
被试数量: 4 (ZFL, MHD, WGL, ZJK)
情绪类型: 悲伤、积极、中性
分析AU数: {len(au_cols)}

2. 各情绪组内被试差异
---------------------
平均变异系数 (CV):
  悲伤: {avg_cv.get('悲伤', 0):.3f}
  积极: {avg_cv.get('积极', 0):.3f}
  中性: {avg_cv.get('中性', 0):.3f}

3. 组内一致性评估
-----------------
"""

for emotion in emotions:
    cv = avg_cv.get(emotion, 0)
    consistency = "高" if cv < 0.3 else "中" if cv < 0.5 else "低"
    report += f"  {emotion}情绪一致性: {consistency} (CV={cv:.3f})\n"

report += """
4. 输出文件
-----------
statistics/
  - subject_emotion_au_means.csv
  - emotion_subject_variability.csv

heatmaps/
  - heatmap_subject_au_悲伤.png
  - heatmap_subject_au_积极.png
  - heatmap_subject_au_中性.png
  - heatmap_group_emotion_comparison.png
  - heatmap_inter_subject_cv.png

================================================================================
"""

with open(f'{output_dir}/analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✓ 患者组分析完成: {output_dir}")
