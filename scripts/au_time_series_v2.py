#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AU时间轨迹分析 - 修正版（使用原始数据，减少平滑）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 文件映射 - 悲伤、中性、积极
EMOTION_FILES = {
    '悲伤': {
        'M1': '/root/.openclaw/media/inbound/file_3---b3314058-964d-470d-8293-13430fdde2c6.csv',
        'M2': '/root/.openclaw/media/inbound/file_4---0dd96eb3-72ff-4ced-a1b8-c5c51fad721a.csv',
        'F1': '/root/.openclaw/media/inbound/file_5---69ad20a2-5a2f-4f18-bdef-056d8c24d515.csv',
        'color': '#3498db'
    },
    '中性': {
        'M1': '/root/.openclaw/media/inbound/file_18---73cb1d9c-9f3c-4f21-917a-ae9408962385.csv',
        'M2': '/root/.openclaw/media/inbound/file_19---476a6dde-2bc6-48b4-89d3-8c3e70cbd0fd.csv',
        'F1': '/root/.openclaw/media/inbound/file_20---333e020a-bdf5-44a5-b833-c3179c272ccc.csv',
        'color': '#95a5a6'
    },
    '积极': {
        'M1': '/root/.openclaw/media/inbound/file_21---c1ecbaad-5700-42b7-a743-1b75f81b7ff1.csv',
        'M2': '/root/.openclaw/media/inbound/file_22---772490a5-e791-43b9-8f4a-25c2f614570a.csv',
        'F1': '/root/.openclaw/media/inbound/file_23---06535c58-c474-473b-a68d-aadcee3e3ca7.csv',
        'color': '#e74c3c'
    }
}

KEY_AUS = ['AU04_r', 'AU07_r', 'AU06_r', 'AU12_r']
AU_NAMES = {
    'AU04_r': 'AU04 (眉毛下垂)',
    'AU07_r': 'AU07 (眼睑收紧)',
    'AU06_r': 'AU06 (脸颊提升)',
    'AU12_r': 'AU12 (嘴角上扬)'
}

# 创建输出目录
timestamp = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = f'/root/.openclaw/workspace/analysis_results/{timestamp}_AU时间轨迹分析_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("AU时间轨迹分析 v2（修正版）")
print("=" * 70)

def load_data(file_path):
    """加载原始数据"""
    df = pd.read_csv(file_path)
    # 去除列名中的空格
    df.columns = df.columns.str.strip()
    if 'timestamp' not in df.columns:
        df['timestamp'] = df['frame'] / 30.0
    return df

# 加载所有数据
print("\n【1. 加载数据】")
all_data = {}
for emotion, files in EMOTION_FILES.items():
    print(f"  {emotion}:")
    all_data[emotion] = {}
    for subject in ['M1', 'M2', 'F1']:
        try:
            df = load_data(files[subject])
            all_data[emotion][subject] = df
            print(f"    {subject}: {len(df)}帧, {df['timestamp'].max():.1f}秒")
        except Exception as e:
            print(f"    {subject}: 错误 - {e}")

# 对M1和M2数据进行子采样（悲伤M1数据太长）
print("\n【2. 数据预处理 - 子采样对齐】")
for emotion in all_data:
    if 'M1' in all_data[emotion] and 'M2' in all_data[emotion]:
        m1_len = len(all_data[emotion]['M1'])
        m2_len = len(all_data[emotion]['M2'])
        
        if m1_len > m2_len * 1.5:  # M1明显更长
            # 对M1进行子采样
            step = m1_len // m2_len
            df_m1 = all_data[emotion]['M1']
            all_data[emotion]['M1'] = df_m1.iloc[::step].reset_index(drop=True)
            all_data[emotion]['M1']['timestamp'] = np.linspace(
                0, df_m1['timestamp'].max(), len(all_data[emotion]['M1'])
            )
            print(f"  {emotion}: M1子采样 {m1_len} -> {len(all_data[emotion]['M1'])}帧")

# ==================== 1. AU07时间轨迹（性别差异之王）====================
print("\n【3. 生成AU07跨情绪时间轨迹】")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, emotion in enumerate(['悲伤', '中性', '积极']):
    ax = axes[idx]
    data = all_data[emotion]
    
    # 男性M1
    if 'M1' in data:
        df = data['M1']
        ax.plot(df['timestamp'], df['AU07_r'], alpha=0.5, color='#2980b9', linewidth=1, label='M1')
    
    # 男性M2
    if 'M2' in data:
        df = data['M2']
        ax.plot(df['timestamp'], df['AU07_r'], alpha=0.5, color='#27ae60', linewidth=1, label='M2')
    
    # 男性平均（重采样到相同长度）
    if 'M1' in data and 'M2' in data:
        m1_df = data['M1']
        m2_df = data['M2']
        min_len = min(len(m1_df), len(m2_df))
        time_axis = np.linspace(0, max(m1_df['timestamp'].iloc[-1], m2_df['timestamp'].iloc[-1]), min_len)
        m1_vals = np.interp(time_axis, m1_df['timestamp'], m1_df['AU07_r'])
        m2_vals = np.interp(time_axis, m2_df['timestamp'], m2_df['AU07_r'])
        male_avg = (m1_vals + m2_vals) / 2
        ax.plot(time_axis, male_avg, color='#3498db', linewidth=2.5, label='男性平均')
    
    # 女性
    if 'F1' in data:
        df = data['F1']
        ax.plot(df['timestamp'], df['AU07_r'], color='#e91e63', linewidth=2.5, label='女性')
    
    ax.set_xlabel('时间 (秒)', fontsize=11)
    ax.set_ylabel('AU07 激活强度', fontsize=11)
    ax.set_title(f'{emotion}情绪 - AU07 (眼睑收紧) 时间轨迹', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/AU07_time_trajectory_cross_emotion.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ AU07跨情绪时间轨迹图已保存")

# ==================== 2. 四个关键AU的积极情绪轨迹 ====================
print("\n【4. 生成积极情绪四AU轨迹】")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

emotion = '积极'
data = all_data[emotion]

# 男性
ax_male = axes[0]
if 'M1' in data and 'M2' in data:
    m1_df = data['M1']
    m2_df = data['M2']
    min_len = min(len(m1_df), len(m2_df))
    time_axis = np.linspace(0, max(m1_df['timestamp'].iloc[-1], m2_df['timestamp'].iloc[-1]), min_len)
    
    for au in KEY_AUS:
        m1_vals = np.interp(time_axis, m1_df['timestamp'], m1_df[au])
        m2_vals = np.interp(time_axis, m2_df['timestamp'], m2_df[au])
        male_avg = (m1_vals + m2_vals) / 2
        ax_male.plot(time_axis, male_avg, label=AU_NAMES[au], linewidth=2)

ax_male.set_ylabel('激活强度', fontsize=11)
ax_male.set_title(f'{emotion}情绪 - 男性平均AU时间轨迹', fontsize=12, fontweight='bold')
ax_male.legend(loc='upper right')
ax_male.grid(alpha=0.3)
ax_male.set_ylim(0, 3)

# 女性
ax_female = axes[1]
if 'F1' in data:
    df = data['F1']
    for au in KEY_AUS:
        ax_female.plot(df['timestamp'], df[au], label=AU_NAMES[au], linewidth=2)

ax_female.set_xlabel('时间 (秒)', fontsize=11)
ax_female.set_ylabel('激活强度', fontsize=11)
ax_female.set_title(f'{emotion}情绪 - 女性AU时间轨迹', fontsize=12, fontweight='bold')
ax_female.legend(loc='upper right')
ax_female.grid(alpha=0.3)
ax_female.set_ylim(0, 3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/{emotion}_four_AU_time_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {emotion}情绪四AU轨迹图已保存")

# ==================== 3. 悲伤情绪四AU轨迹（对比最强烈）====================
print("\n【5. 生成悲伤情绪四AU轨迹】")

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

emotion = '悲伤'
data = all_data[emotion]

# 男性
ax_male = axes[0]
if 'M1' in data and 'M2' in data:
    m1_df = data['M1']
    m2_df = data['M2']
    min_len = min(len(m1_df), len(m2_df))
    time_axis = np.linspace(0, max(m1_df['timestamp'].iloc[-1], m2_df['timestamp'].iloc[-1]), min_len)
    
    for au in KEY_AUS:
        m1_vals = np.interp(time_axis, m1_df['timestamp'], m1_df[au])
        m2_vals = np.interp(time_axis, m2_df['timestamp'], m2_df[au])
        male_avg = (m1_vals + m2_vals) / 2
        ax_male.plot(time_axis, male_avg, label=AU_NAMES[au], linewidth=2)

ax_male.set_ylabel('激活强度', fontsize=11)
ax_male.set_title(f'{emotion}情绪 - 男性平均AU时间轨迹', fontsize=12, fontweight='bold')
ax_male.legend(loc='upper right')
ax_male.grid(alpha=0.3)
ax_male.set_ylim(0, 3)

# 女性
ax_female = axes[1]
if 'F1' in data:
    df = data['F1']
    for au in KEY_AUS:
        ax_female.plot(df['timestamp'], df[au], label=AU_NAMES[au], linewidth=2)

ax_female.set_xlabel('时间 (秒)', fontsize=11)
ax_female.set_ylabel('激活强度', fontsize=11)
ax_female.set_title(f'{emotion}情绪 - 女性AU时间轨迹', fontsize=12, fontweight='bold')
ax_female.legend(loc='upper right')
ax_female.grid(alpha=0.3)
ax_female.set_ylim(0, 3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/{emotion}_four_AU_time_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ {emotion}情绪四AU轨迹图已保存")

# ==================== 4. 单AU三情绪对比 ====================
print("\n【6. 生成单AU三情绪对比轨迹】")

for au in KEY_AUS:
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for emotion in ['悲伤', '中性', '积极']:
        data = all_data[emotion]
        color = {'悲伤': '#3498db', '中性': '#95a5a6', '积极': '#e74c3c'}[emotion]
        
        # 男性平均
        if 'M1' in data and 'M2' in data:
            m1_df = data['M1']
            m2_df = data['M2']
            min_len = min(len(m1_df), len(m2_df))
            time_axis = np.linspace(0, 60, min_len)  # 标准化到60秒
            m1_vals = np.interp(time_axis, m1_df['timestamp'], m1_df[au])
            m2_vals = np.interp(time_axis, m2_df['timestamp'], m2_df[au])
            male_avg = (m1_vals + m2_vals) / 2
            ax.plot(time_axis, male_avg, label=f'{emotion}-男', color=color, linewidth=2, alpha=0.8)
        
        # 女性
        if 'F1' in data:
            df = data['F1']
            time_axis_f = np.linspace(0, 60, len(df))
            ax.plot(time_axis_f, df[au], label=f'{emotion}-女', color=color, linewidth=2, linestyle='--', alpha=0.8)
    
    ax.set_xlabel('时间 (秒)', fontsize=11)
    ax.set_ylabel(f'{au} 激活强度', fontsize=11)
    ax.set_title(f'{AU_NAMES[au]} 跨情绪时间轨迹对比', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{au}_three_emotion_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

print(f"  ✓ 4个AU的三情绪对比轨迹图已保存")

# ==================== 5. AU07性别差异时间分布 ====================
print("\n【7. 生成AU07性别差异时间分布】")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, emotion in enumerate(['悲伤', '中性', '积极']):
    ax = axes[idx]
    data = all_data[emotion]
    
    if 'M1' in data and 'M2' in data and 'F1' in data:
        # 统一时间轴（0-60秒）
        time_common = np.linspace(0, 60, 500)
        
        m1_df = data['M1']
        m2_df = data['M2']
        f1_df = data['F1']
        
        # 插值到统一时间轴
        m1_vals = np.interp(time_common, m1_df['timestamp'], m1_df['AU07_r'])
        m2_vals = np.interp(time_common, m2_df['timestamp'], m2_df['AU07_r'])
        male_avg = (m1_vals + m2_vals) / 2
        female_vals = np.interp(time_common, f1_df['timestamp'], f1_df['AU07_r'])
        
        gender_diff = male_avg - female_vals
        
        # 绘制
        ax.fill_between(time_common, 0, gender_diff, 
                       where=(gender_diff > 0), color='#3498db', alpha=0.5, label='男性>女性')
        ax.fill_between(time_common, 0, gender_diff, 
                       where=(gender_diff <= 0), color='#e91e63', alpha=0.5, label='女性>男性')
        ax.plot(time_common, gender_diff, color='#2c3e50', linewidth=1)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('时间 (秒)', fontsize=11)
        ax.set_ylabel('性别差异 (男-女)', fontsize=11)
        ax.set_title(f'{emotion}情绪 - AU07性别差异时间分布', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        
        mean_diff = np.mean(gender_diff)
        ax.text(0.02, 0.95, f'平均差异: {mean_diff:.2f}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/AU07_gender_diff_time_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ AU07性别差异时间分布图已保存")

# ==================== 6. AU06+AU12微笑协同散点图 ====================
print("\n【8. 生成AU06+AU12协同散点图】")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, emotion in enumerate(['悲伤', '中性', '积极']):
    ax = axes[idx]
    data = all_data[emotion]
    
    if 'M1' in data and 'M2' in data and 'F1' in data:
        # 男性数据（合并M1和M2）
        for subject in ['M1', 'M2']:
            df = data[subject]
            ax.scatter(df['AU06_r'], df['AU12_r'], alpha=0.3, color='#3498db', s=10)
        
        # 女性数据
        df_f = data['F1']
        ax.scatter(df_f['AU06_r'], df_f['AU12_r'], alpha=0.3, color='#e91e63', s=10)
        
        # 参考线
        ax.plot([0, 3], [0, 3], 'k--', alpha=0.3, label='AU06=AU12')
        
        ax.set_xlabel('AU06 (脸颊提升)', fontsize=11)
        ax.set_ylabel('AU12 (嘴角上扬)', fontsize=11)
        ax.set_title(f'{emotion}情绪', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.1, 3)
        ax.set_ylim(-0.1, 3)
        
        # 添加图例说明
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='男性', alpha=0.7),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e91e63', markersize=8, label='女性', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='upper left')

plt.suptitle('AU06 vs AU12 协同散点图 (杜氏微笑分析)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/AU06_AU12_smile_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ AU06+AU12协同散点图已保存")

print("\n" + "=" * 70)
print("AU时间轨迹分析 v2 完成！")
print(f"输出目录: {OUTPUT_DIR}")
print("\n生成的文件:")
print("  📈 AU07跨情绪时间轨迹")
print("  📊 悲伤/积极情绪四AU轨迹")
print("  📉 4个AU三情绪对比轨迹")
print("  🔍 AU07性别差异时间分布")
print("  🎯 AU06+AU12微笑协同散点图")
print("=" * 70)
