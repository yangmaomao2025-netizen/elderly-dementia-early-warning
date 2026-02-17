#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
积极情绪AU数据分析 - 简化版
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置
FILE_MAPPING = {
    '/root/.openclaw/media/inbound/file_21---c1ecbaad-5700-42b7-a743-1b75f81b7ff1.csv': ('M1', 'Male'),
    '/root/.openclaw/media/inbound/file_22---772490a5-e791-43b9-8f4a-25c2f614570a.csv': ('M2', 'Male'),
    '/root/.openclaw/media/inbound/file_23---06535c58-c474-473b-a68d-aadcee3e3ca7.csv': ('F1', 'Female'),
}

AU_COLUMNS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

AU_NAMES = {
    'AU01_r': 'AU01(眉毛内侧上扬)',
    'AU02_r': 'AU02(眉毛外侧上扬)',
    'AU04_r': 'AU04(眉毛下垂)',
    'AU05_r': 'AU05(上眼睑上扬)',
    'AU06_r': 'AU06(脸颊上扬)',
    'AU07_r': 'AU07(眼睑紧绷)',
    'AU09_r': 'AU09(鼻子皱起)',
    'AU10_r': 'AU10(上唇上扬)',
    'AU12_r': 'AU12(嘴角上扬)',
    'AU14_r': 'AU14(酒窝)',
    'AU15_r': 'AU15(嘴角下垂)',
    'AU17_r': 'AU17(下巴上扬)',
    'AU20_r': 'AU20(嘴唇横向伸展)',
    'AU23_r': 'AU23(嘴唇收紧)',
    'AU25_r': 'AU25(嘴唇分开)',
    'AU26_r': 'AU26(下颌下垂)',
    'AU45_r': 'AU45(眨眼)',
}

# 创建输出目录
today = datetime.now().strftime('%Y-%m-%d')
base_dir = f"/root/.openclaw/workspace/analysis_results/{today}_积极情绪_性别对比"
dirs = {d: os.path.join(base_dir, d) for d in ['heatmaps', 'barplots', 'statistics', 'raw_data']}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)

print(f"📁 输出目录: {base_dir}")

# 加载数据
print("\n📂 加载数据...")
data = {}
for filepath, (subject_id, gender) in FILE_MAPPING.items():
    print(f"  加载 {subject_id}...", end=" ")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df[df['confidence'] > 0.8].reset_index(drop=True)
    data[subject_id] = {'df': df, 'gender': gender}
    print(f"✓ ({len(df)} 帧)")

# 1. 生成热力图
print("\n📊 生成AU激活热力图...")
for subject_id, info in data.items():
    print(f"  处理 {subject_id}...", end=" ")
    df = info['df']
    gender = info['gender']
    
    window_size = 100
    n_windows = len(df) // window_size
    
    heatmap_data = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_mean = df.iloc[start:end][AU_COLUMNS].mean().values
        heatmap_data.append(window_mean)
    
    heatmap_data = np.array(heatmap_data).T
    
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=2.5)
    ax.set_yticks(range(len(AU_COLUMNS)))
    ax.set_yticklabels([AU_NAMES[au] for au in AU_COLUMNS], fontsize=8)
    ax.set_xlabel('时间段 (约3秒/格)', fontsize=12)
    ax.set_title(f'积极情绪 - {subject_id} ({gender}) - AU激活强度热力图', fontsize=14)
    plt.colorbar(im, ax=ax, label='AU强度')
    
    save_path = f"{dirs['heatmaps']}/{subject_id}_heatmap.png"
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"✓")

# 2. 计算统计数据
print("\n📊 计算统计数据...")
subject_means = {}
for subject_id, info in data.items():
    df = info['df']
    subject_means[subject_id] = [df[au].mean() for au in AU_COLUMNS]

# 男性平均
male_mean = [(subject_means['M1'][i] + subject_means['M2'][i]) / 2 for i in range(len(AU_COLUMNS))]
female_mean = subject_means['F1']
male_diff = [abs(subject_means['M1'][i] - subject_means['M2'][i]) for i in range(len(AU_COLUMNS))]

# 3. 生成性别对比柱状图
print("\n📊 生成性别对比柱状图...")
x = np.arange(len(AU_COLUMNS))
width = 0.35

fig, ax = plt.subplots(figsize=(16, 7))
ax.bar(x - width/2, male_mean, width, label='男性平均 (M1+M2)/2', color='#3498db', alpha=0.8)
ax.bar(x + width/2, female_mean, width, label='女性 (F1)', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Action Units', fontsize=12)
ax.set_ylabel('平均激活强度', fontsize=12)
ax.set_title('积极情绪 - 性别AU激活对比', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels([au.replace('_r', '') for au in AU_COLUMNS], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{dirs['barplots']}/gender_comparison_barplot.png", dpi=150)
plt.close()
print(f"  ✓ 柱状图已保存")

# 4. 导出数据
print("\n📊 导出统计数据...")
gender_df = pd.DataFrame({
    'AU': [au.replace('_r', '') for au in AU_COLUMNS],
    'Male_M1': subject_means['M1'],
    'Male_M2': subject_means['M2'],
    'Male_Avg': male_mean,
    'Female_F1': female_mean,
    'Male_Internal_Diff': male_diff,
    'Gender_Diff(M-F)': [m - f for m, f in zip(male_mean, female_mean)],
})
gender_df.to_csv(f"{dirs['raw_data']}/gender_comparison.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ 数据已导出")

# 5. 生成报告
print("\n📊 生成分析报告...")
gender_diffs = [(AU_COLUMNS[i], male_mean[i] - female_mean[i], male_mean[i], female_mean[i]) 
                for i in range(len(AU_COLUMNS))]
gender_diffs.sort(key=lambda x: abs(x[1]), reverse=True)

report = []
report.append("=" * 80)
report.append("积极情绪AU数据 - 统计分析报告")
report.append("=" * 80)
report.append("")
report.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"被试数量: 3人 (男性2人, 女性1人)")
report.append(f"情绪类型: 积极情绪 (Positive/Happy)")
report.append("")

report.append("【1. 个体内AU激活均值 Top 5】")
for subject_id, info in data.items():
    df = info['df']
    report.append(f"\n{subject_id} ({info['gender']}):")
    au_means = [(au, df[au].mean()) for au in AU_COLUMNS]
    au_means.sort(key=lambda x: x[1], reverse=True)
    for au, mean_val in au_means[:5]:
        report.append(f"  {au}: {mean_val:.3f}")

report.append("\n【2. 性别差异 (男性平均 - 女性)】")
for au, diff, m_val, f_val in gender_diffs:
    direction = "男性>女性" if diff > 0 else "女性>男性"
    report.append(f"  {au}: {diff:+.3f} (男:{m_val:.3f}, 女:{f_val:.3f}) [{direction}]")

report.append("\n【3. 关键发现】")
top_diff = gender_diffs[0]
report.append(f"• 最大性别差异AU: {top_diff[0]} (差异={top_diff[1]:.3f})")

# 检查女性为0的AU
zero_aus = [AU_COLUMNS[i] for i in range(len(AU_COLUMNS)) if female_mean[i] == 0]
if zero_aus:
    report.append(f"• 女性无激活AU: {', '.join(zero_aus)}")

# 积极情绪特有
au12_idx = AU_COLUMNS.index('AU12_r')
au06_idx = AU_COLUMNS.index('AU06_r')
report.append(f"• AU12 (微笑标志): 男性={male_mean[au12_idx]:.3f}, 女性={female_mean[au12_idx]:.3f}")
report.append(f"• AU06 (脸颊上扬): 男性={male_mean[au06_idx]:.3f}, 女性={female_mean[au06_idx]:.3f}")

report.append("")
report.append("=" * 80)

report_text = "\n".join(report)
with open(f"{dirs['statistics']}/analysis_report.txt", 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)

print(f"\n✅ 分析完成！结果保存在: {base_dir}")
