#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AU时间轨迹分析 - 跨情绪时间序列可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 文件映射
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
OUTPUT_DIR = f'/root/.openclaw/workspace/analysis_results/{timestamp}_AU时间轨迹分析'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("AU时间轨迹分析")
print("=" * 70)

def load_and_process(file_path, au_list):
    """加载数据并提取关键AU的时间序列"""
    df = pd.read_csv(file_path)
    
    # 提取时间（秒）
    if 'timestamp' in df.columns:
        time_col = 'timestamp'
    else:
        # 如果没有timestamp，用frame/30计算
        df['timestamp'] = df['frame'] / 30.0
        time_col = 'timestamp'
    
    result = {'time': df[time_col].values}
    for au in au_list:
        if au in df.columns:
            # 应用平滑（移动平均窗口=10帧）
            smoothed = uniform_filter1d(df[au].values, size=10)
            result[au] = smoothed
        else:
            result[au] = np.zeros(len(df))
    
    return pd.DataFrame(result)

# 加载所有数据
print("\n【1. 加载数据】")
all_data = {}
for emotion, files in EMOTION_FILES.items():
    print(f"  {emotion}:")
    all_data[emotion] = {}
    for subject, path in files.items():
        if subject != 'color':
            try:
                df = load_and_process(path, KEY_AUS)
                all_data[emotion][subject] = df
                print(f"    {subject}: {len(df)}帧, {df['time'].max():.1f}秒")
            except Exception as e:
                print(f"    {subject}: 加载失败 - {e}")

# ==================== 2. 生成单AU跨情绪时间轨迹图 ====================
print("\n【2. 生成单AU跨情绪时间轨迹图】")

for au in KEY_AUS:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    for idx, (emotion, data) in enumerate(all_data.items()):
        ax = axes[idx]
        
        # 男性平均
        if 'M1' in data and 'M2' in data:
            m1_df = data['M1']
            m2_df = data['M2']
            # 对齐时间轴（取最短）
            min_len = min(len(m1_df), len(m2_df))
            time_axis = m1_df['time'].values[:min_len]
            male_avg = (m1_df[au].values[:min_len] + m2_df[au].values[:min_len]) / 2
            
            ax.plot(time_axis, male_avg, label='男性平均', color='#3498db', linewidth=2)
        
        # 女性
        if 'F1' in data:
            f1_df = data['F1']
            time_axis_f = f1_df['time'].values
            ax.plot(time_axis_f, f1_df[au].values, label='女性', color='#e91e63', linewidth=2)
        
        ax.set_xlabel('时间 (秒)', fontsize=11)
        ax.set_ylabel(f'{au} 激活强度', fontsize=11)
        ax.set_title(f'{emotion}情绪 - {AU_NAMES[au]}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, max(time_axis[-1] if 'time_axis' in locals() else 60, 
                         time_axis_f[-1] if 'time_axis_f' in locals() else 60))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{au}_cross_emotion_time_series.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {au} 时间轨迹图已保存")

# ==================== 3. 生成单情绪四AU对比图 ====================
print("\n【3. 生成单情绪四AU对比图】")

for emotion, data in all_data.items():
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # 上：男性
    ax_male = axes[0]
    if 'M1' in data and 'M2' in data:
        m1_df = data['M1']
        m2_df = data['M2']
        min_len = min(len(m1_df), len(m2_df))
        time_axis = m1_df['time'].values[:min_len]
        
        for au in KEY_AUS:
            male_avg = (m1_df[au].values[:min_len] + m2_df[au].values[:min_len]) / 2
            ax_male.plot(time_axis, male_avg, label=AU_NAMES[au], linewidth=2)
        
        ax_male.set_ylabel('激活强度', fontsize=11)
        ax_male.set_title(f'{emotion}情绪 - 男性平均AU时间轨迹', fontsize=12, fontweight='bold')
        ax_male.legend(loc='upper right')
        ax_male.grid(alpha=0.3)
    
    # 下：女性
    ax_female = axes[1]
    if 'F1' in data:
        f1_df = data['F1']
        time_axis_f = f1_df['time'].values
        
        for au in KEY_AUS:
            ax_female.plot(time_axis_f, f1_df[au].values, label=AU_NAMES[au], linewidth=2)
        
        ax_female.set_xlabel('时间 (秒)', fontsize=11)
        ax_female.set_ylabel('激活强度', fontsize=11)
        ax_female.set_title(f'{emotion}情绪 - 女性AU时间轨迹', fontsize=12, fontweight='bold')
        ax_female.legend(loc='upper right')
        ax_female.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{emotion}_four_au_time_series.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {emotion}情绪四AU对比图已保存")

# ==================== 4. 生成AU07专题分析图（性别差异之王） ====================
print("\n【4. 生成AU07专题时间分析】")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for idx, (emotion, data) in enumerate(all_data.items()):
    ax = axes[idx]
    
    # 计算性别差异时间序列
    if 'M1' in data and 'M2' in data and 'F1' in data:
        m1_df = data['M1']
        m2_df = data['M2']
        f1_df = data['F1']
        
        # 对齐长度
        min_len = min(len(m1_df), len(m2_df), len(f1_df))
        time_axis = m1_df['time'].values[:min_len]
        male_avg = (m1_df['AU07_r'].values[:min_len] + m2_df['AU07_r'].values[:min_len]) / 2
        female_val = f1_df['AU07_r'].values[:min_len]
        gender_diff = male_avg - female_val
        
        # 绘制
        ax.fill_between(time_axis, 0, gender_diff, 
                       where=(gender_diff > 0), color='#3498db', alpha=0.5, label='男性>女性')
        ax.fill_between(time_axis, 0, gender_diff, 
                       where=(gender_diff <= 0), color='#e91e63', alpha=0.5, label='女性>男性')
        ax.plot(time_axis, gender_diff, color='#2c3e50', linewidth=1.5)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('时间 (秒)', fontsize=11)
        ax.set_ylabel('性别差异 (男-女)', fontsize=11)
        ax.set_title(f'{emotion}情绪 - AU07性别差异时间分布', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        
        # 标注统计信息
        mean_diff = np.mean(gender_diff)
        ax.text(0.02, 0.95, f'平均差异: {mean_diff:.2f}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/AU07_gender_diff_time_series.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ AU07性别差异时间分布图已保存")

# ==================== 5. 生成AU协同分析图（AU06+AU12微笑组合） ====================
print("\n【5. 生成AU06+AU12微笑协同分析】")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (emotion, data) in enumerate(all_data.items()):
    ax = axes[idx]
    
    if 'M1' in data and 'M2' in data and 'F1' in data:
        m1_df = data['M1']
        m2_df = data['M2']
        f1_df = data['F1']
        
        min_len = min(len(m1_df), len(m2_df))
        time_axis = m1_df['time'].values[:min_len]
        
        # 男性：AU06 vs AU12
        male_au06 = (m1_df['AU06_r'].values[:min_len] + m2_df['AU06_r'].values[:min_len]) / 2
        male_au12 = (m1_df['AU12_r'].values[:min_len] + m2_df['AU12_r'].values[:min_len]) / 2
        
        # 女性：AU06 vs AU12
        female_au06 = f1_df['AU06_r'].values[:min_len]
        female_au12 = f1_df['AU12_r'].values[:min_len]
        
        # 散点图
        ax.scatter(male_au06, male_au12, alpha=0.3, label='男性', color='#3498db', s=20)
        ax.scatter(female_au06, female_au12, alpha=0.3, label='女性', color='#e91e63', s=20)
        
        ax.plot([0, 2], [0, 2], 'k--', alpha=0.3, label='AU06=AU12线')
        ax.set_xlabel('AU06 (脸颊提升)', fontsize=11)
        ax.set_ylabel('AU12 (嘴角上扬)', fontsize=11)
        ax.set_title(f'{emotion}情绪', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.1, 2.5)
        ax.set_ylim(-0.1, 2.5)

plt.suptitle('AU06 vs AU12 协同散点图 (杜氏微笑分析)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/AU06_AU12_correlation_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ AU06+AU12协同散点图已保存")

# ==================== 6. 生成激活峰值分析 ====================
print("\n【6. 生成激活峰值时间分布】")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for au_idx, au in enumerate(KEY_AUS):
    ax = axes[au_idx]
    
    peak_data = {'悲伤': {'male': [], 'female': []},
                '中性': {'male': [], 'female': []},
                '积极': {'male': [], 'female': []}}
    
    for emotion, data in all_data.items():
        if 'M1' in data and 'M2' in data and 'F1' in data:
            m1_df = data['M1']
            m2_df = data['M2']
            f1_df = data['F1']
            
            # 找峰值（局部最大值）
            from scipy.signal import find_peaks
            
            # 男性平均
            min_len = min(len(m1_df), len(m2_df))
            male_avg = (m1_df[au].values[:min_len] + m2_df[au].values[:min_len]) / 2
            time_axis = m1_df['time'].values[:min_len]
            
            peaks_m, _ = find_peaks(male_avg, height=0.5, distance=30)
            peak_data[emotion]['male'] = time_axis[peaks_m]
            
            # 女性
            time_axis_f = f1_df['time'].values
            peaks_f, _ = find_peaks(f1_df[au].values, height=0.5, distance=30)
            peak_data[emotion]['female'] = time_axis_f[peaks_f]
    
    # 绘制直方图
    bins = np.linspace(0, 60, 20)
    bottom = np.zeros(len(bins)-1)
    
    emotions_list = ['悲伤', '中性', '积极']
    colors = ['#3498db', '#95a5a6', '#e74c3c']
    
    for emotion, color in zip(emotions_list, colors):
        if len(peak_data[emotion]['male']) > 0:
            counts, _ = np.histogram(peak_data[emotion]['male'], bins=bins)
            ax.bar(bins[:-1], counts, width=bins[1]-bins[0], bottom=bottom, 
                  label=f'{emotion}-男', color=color, alpha=0.7, edgecolor='black')
            bottom += counts
    
    ax.set_xlabel('时间 (秒)', fontsize=11)
    ax.set_ylabel('峰值数量', fontsize=11)
    ax.set_title(f'{AU_NAMES[au]} 峰值时间分布', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/peak_time_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ 激活峰值时间分布图已保存")

print("\n" + "=" * 70)
print("AU时间轨迹分析完成！")
print(f"输出目录: {OUTPUT_DIR}")
print("\n生成的文件:")
print("  📈 单AU跨情绪时间轨迹图 (4个)")
print("  📊 单情绪四AU对比图 (3个)")
print("  🔍 AU07性别差异时间分布")
print("  🎯 AU06+AU12微笑协同散点图")
print("  📊 激活峰值时间分布")
print("=" * 70)
