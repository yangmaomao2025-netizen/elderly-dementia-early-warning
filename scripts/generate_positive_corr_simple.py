#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
积极情绪AU相关性热力图 - 简化版
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

AU_SHORT_NAMES = [au.replace('_r', '') for au in AU_COLUMNS]

# 创建输出目录
today = datetime.now().strftime('%Y-%m-%d')
base_dir = f"/root/.openclaw/workspace/analysis_results/{today}_积极情绪_性别对比"
corr_dir = os.path.join(base_dir, 'correlation_matrix')
os.makedirs(corr_dir, exist_ok=True)

print(f"📁 输出目录: {corr_dir}")

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

# 存储相关性矩阵
all_corr_matrices = {}

# 生成个人相关性热力图
print("\n📊 生成AU相关性热力图...")
for subject_id, info in data.items():
    print(f"  处理 {subject_id}...", end=" ")
    df = info['df']
    gender = info['gender']
    
    # 计算相关性矩阵
    au_data = df[AU_COLUMNS].copy()
    corr_matrix = au_data.corr()
    
    # 检查常数列
    constant_cols = []
    for col in AU_COLUMNS:
        if df[col].std() == 0 or df[col].nunique() == 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"\n    警告: 以下AU无变化: {[c.replace('_r', '') for c in constant_cols]}")
    
    # 填充NaN为0
    corr_matrix = corr_matrix.fillna(0)
    all_corr_matrices[subject_id] = corr_matrix
    
    # 生成热力图 - 只显示下三角
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    im = ax.imshow(np.ma.array(corr_matrix.values, mask=mask), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # 添加数值标注
    for i in range(len(AU_COLUMNS)):
        for j in range(i+1):
            val = corr_matrix.iloc[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=text_color)
    
    ax.set_xticks(range(len(AU_COLUMNS)))
    ax.set_yticks(range(len(AU_COLUMNS)))
    ax.set_xticklabels(AU_SHORT_NAMES, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(AU_SHORT_NAMES, fontsize=10)
    ax.set_title(f'积极情绪 - {subject_id} ({gender}) - AU相关性矩阵', fontsize=14)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    
    save_path = os.path.join(corr_dir, f'{subject_id}_correlation.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓")

# 生成性别平均相关性
print("\n📊 生成性别平均相关性...")

# 男性平均
male_corr = (all_corr_matrices['M1'] + all_corr_matrices['M2']) / 2
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(male_corr, dtype=bool), k=1)
im = ax.imshow(np.ma.array(male_corr.values, mask=mask), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

for i in range(len(AU_COLUMNS)):
    for j in range(i+1):
        val = male_corr.iloc[i, j]
        text_color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=text_color)

ax.set_xticks(range(len(AU_COLUMNS)))
ax.set_yticks(range(len(AU_COLUMNS)))
ax.set_xticklabels(AU_SHORT_NAMES, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(AU_SHORT_NAMES, fontsize=10)
ax.set_title('积极情绪 - 男性平均 (M1+M2)/2 - AU相关性矩阵', fontsize=14)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(corr_dir, 'Male_Avg_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 男性平均相关性")

# 女性
female_corr = all_corr_matrices['F1']
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(female_corr, dtype=bool), k=1)
im = ax.imshow(np.ma.array(female_corr.values, mask=mask), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

for i in range(len(AU_COLUMNS)):
    for j in range(i+1):
        val = female_corr.iloc[i, j]
        text_color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=text_color)

ax.set_xticks(range(len(AU_COLUMNS)))
ax.set_yticks(range(len(AU_COLUMNS)))
ax.set_xticklabels(AU_SHORT_NAMES, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(AU_SHORT_NAMES, fontsize=10)
ax.set_title('积极情绪 - 女性 (F1) - AU相关性矩阵', fontsize=14)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(corr_dir, 'Female_F1_correlation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 女性相关性")

# 生成分析报告
print("\n📊 生成分析报告...")
report = []
report.append("=" * 80)
report.append("积极情绪AU相关性分析报告")
report.append("=" * 80)
report.append("")
report.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append("")

for subject_id in ['M1', 'M2', 'F1']:
    gender = data[subject_id]['gender']
    corr_matrix = all_corr_matrices[subject_id]
    
    report.append(f"【{subject_id} ({gender})】")
    
    # 找出强相关对 (|r| >= 0.5)
    strong_corrs = []
    for i in range(len(AU_COLUMNS)):
        for j in range(i+1, len(AU_COLUMNS)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= 0.5:
                strong_corrs.append({
                    'AU1': AU_SHORT_NAMES[i],
                    'AU2': AU_SHORT_NAMES[j],
                    'corr': corr_val
                })
    
    strong_corrs.sort(key=lambda x: abs(x['corr']), reverse=True)
    
    if strong_corrs:
        report.append(f"  强相关AU对 (|r| >= 0.5): 共{len(strong_corrs)}对")
        for i, sc in enumerate(strong_corrs[:10]):
            direction = "正相关" if sc['corr'] > 0 else "负相关"
            report.append(f"    {sc['AU1']} - {sc['AU2']}: r={sc['corr']:.3f} ({direction})")
        if len(strong_corrs) > 10:
            report.append(f"    ... 还有 {len(strong_corrs)-10} 对")
    else:
        report.append("  无强相关AU对 (|r| < 0.5)")
    report.append("")

report.append("=" * 80)

report_text = "\n".join(report)
with open(os.path.join(corr_dir, 'correlation_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)

print(f"\n✅ 相关性分析完成！结果保存在: {corr_dir}")

# 列出生成的文件
print("\n📋 生成的文件:")
for f in sorted(os.listdir(corr_dir)):
    print(f"  - {f}")
