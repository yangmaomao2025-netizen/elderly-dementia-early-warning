#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中性情绪个体内差异与性别比较分析 - 参照悲伤情绪分析标准
"""

import csv
import os
import math
import shutil
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 中性情绪数据文件映射 (2男 + 1女)
SUBJECTS = {
    'file_18---73cb1d9c-9f3c-4f21-917a-ae9408962385.csv': {'id': 'M1', 'gender': 'male', 'name': '男性1'},
    'file_19---476a6dde-2bc6-48b4-89d3-8c3e70cbd0fd.csv': {'id': 'M2', 'gender': 'male', 'name': '男性2'},
    'file_20---333e020a-bdf5-44a5-b833-c3179c272ccc.csv': {'id': 'F1', 'gender': 'female', 'name': '女性1'}
}

KEY_AUS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
           'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
           'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

AU_NAMES = {
    'AU01_r': '内侧眉毛上扬', 'AU02_r': '外侧眉毛上扬', 'AU04_r': '眉毛下垂',
    'AU05_r': '上眼睑上扬', 'AU06_r': '脸颊上扬', 'AU07_r': '眼睑收紧',
    'AU09_r': '鼻子起皱', 'AU10_r': '上唇上扬', 'AU12_r': '嘴角上扬',
    'AU14_r': '嘴角收紧', 'AU15_r': '嘴角下垂', 'AU17_r': '下巴上扬',
    'AU20_r': '嘴唇伸展', 'AU23_r': '嘴唇收紧', 'AU25_r': '嘴唇分开',
    'AU26_r': '下巴下降', 'AU45_r': '眨眼'
}

def read_csv(filepath):
    """读取CSV并提取AU数据"""
    data = defaultdict(list)
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for raw_key in row.keys():
                key = raw_key.strip()
                if key in KEY_AUS:
                    try:
                        val = float(row[raw_key])
                        data[key].append(val)
                    except:
                        pass
    return data

def calc_stats(values):
    if not values:
        return {'mean': 0, 'std': 0, 'n': 0, 'median': 0, 'min': 0, 'max': 0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0
    std = math.sqrt(variance)
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    return {'mean': mean, 'std': std, 'n': n, 'median': median, 'min': min(values), 'max': max(values)}

def create_subject_heatmap(data, stats, subject_name, output_path):
    """为单个被试创建热力图"""
    fig, ax = plt.subplots(figsize=(6, 12))
    
    # 构建数据矩阵 (17 AU × 1 被试)
    matrix = [[stats[au]['mean']] for au in KEY_AUS]
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=2)
    
    # 设置标签
    ax.set_xticks([0])
    ax.set_xticklabels([subject_name], fontsize=14)
    
    y_labels = [f"{au.replace('_r', '')}" for au in KEY_AUS]
    ax.set_yticks(range(len(KEY_AUS)))
    ax.set_yticklabels(y_labels, fontsize=11)
    
    # 添加数值标注
    for i, au in enumerate(KEY_AUS):
        val = stats[au]['mean']
        ax.text(0, i, f'{val:.2f}', ha="center", va="center", 
                color="white" if val > 1.0 else "black", fontsize=10, fontweight='bold')
    
    ax.set_title(f'中性情绪 - {subject_name} AU激活模式', fontsize=14, pad=15)
    plt.colorbar(im, ax=ax, label='AU Intensity')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_comparison_heatmap(results, output_path):
    """创建3被试对比热力图"""
    fig, ax = plt.subplots(figsize=(8, 12))
    
    # 构建矩阵 (17 AU × 3 被试)
    subjects = ['M1', 'M2', 'F1']
    matrix = [[results[s]['stats'][au]['mean'] for s in subjects] for au in KEY_AUS]
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=2)
    
    # 设置标签
    ax.set_xticks(range(3))
    ax.set_xticklabels([results[s]['info']['name'] for s in subjects], fontsize=13)
    
    y_labels = [f"{au.replace('_r', '')}" for au in KEY_AUS]
    ax.set_yticks(range(len(KEY_AUS)))
    ax.set_yticklabels(y_labels, fontsize=11)
    
    # 添加数值标注
    for i in range(len(KEY_AUS)):
        for j in range(3):
            val = matrix[i][j]
            ax.text(j, i, f'{val:.2f}', ha="center", va="center", 
                    color="white" if val > 1.0 else "black", fontsize=9)
    
    ax.set_title('中性情绪 - 3被试AU激活对比', fontsize=14, pad=15)
    plt.colorbar(im, ax=ax, label='AU Intensity (Mean)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_gender_diff_heatmap(results, output_path):
    """创建性别差异热力图 (男性均值 - 女性)"""
    fig, ax = plt.subplots(figsize=(6, 12))
    
    # 计算性别差异
    m1 = results['M1']['stats']
    m2 = results['M2']['stats']
    f1 = results['F1']['stats']
    
    matrix = [[(m1[au]['mean'] + m2[au]['mean'])/2 - f1[au]['mean']] for au in KEY_AUS]
    
    # 使用 diverging colormap
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    
    ax.set_xticks([0])
    ax.set_xticklabels(['性别差\n(男-女)'], fontsize=12)
    
    y_labels = [f"{au.replace('_r', '')}" for au in KEY_AUS]
    ax.set_yticks(range(len(KEY_AUS)))
    ax.set_yticklabels(y_labels, fontsize=11)
    
    # 添加数值标注
    for i, au in enumerate(KEY_AUS):
        val = matrix[i][0]
        color = "white" if abs(val) > 0.5 else "black"
        ax.text(0, i, f'{val:+.2f}', ha="center", va="center", color=color, fontsize=10)
    
    ax.set_title('中性情绪 - 性别差异热力图', fontsize=14, pad=15)
    plt.colorbar(im, ax=ax, label='Difference (Male - Female)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_barplots(results, output_dir):
    """创建柱状图"""
    # 1. 3被试AU均值对比
    fig, ax = plt.subplots(figsize=(16, 6))
    
    x = np.arange(len(KEY_AUS))
    width = 0.25
    
    colors = {'M1': '#3498db', 'M2': '#2ecc71', 'F1': '#e74c3c'}
    
    for i, subj in enumerate(['M1', 'M2', 'F1']):
        values = [results[subj]['stats'][au]['mean'] for au in KEY_AUS]
        offset = width * (i - 1)
        ax.bar(x + offset, values, width, label=results[subj]['info']['name'], 
               color=colors[subj], alpha=0.8)
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title('中性情绪 - 3被试AU均值对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([au.replace('_r', '') for au in KEY_AUS], fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/barplot_3subjects_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 性别对比柱状图
    fig, ax = plt.subplots(figsize=(14, 6))
    
    male_means = [(results['M1']['stats'][au]['mean'] + results['M2']['stats'][au]['mean'])/2 
                  for au in KEY_AUS]
    female_means = [results['F1']['stats'][au]['mean'] for au in KEY_AUS]
    
    x = np.arange(len(KEY_AUS))
    width = 0.35
    
    ax.bar(x - width/2, male_means, width, label='男性 (n=2)', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, female_means, width, label='女性 (n=1)', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title('中性情绪 - 性别AU均值对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([au.replace('_r', '') for au in KEY_AUS], fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/barplot_gender_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_boxplots(results, output_dir):
    """创建箱线图"""
    key_aus = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU15_r', 'AU17_r']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, au in enumerate(key_aus):
        ax = axes[idx]
        
        data_to_plot = []
        labels = []
        colors = []
        
        for subj in ['M1', 'M2', 'F1']:
            if au in results[subj]['raw'] and results[subj]['raw'][au]:
                data_to_plot.append(results[subj]['raw'][au])
                labels.append(results[subj]['info']['name'])
                colors.append('#3498db' if results[subj]['info']['gender'] == 'male' else '#e74c3c')
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax.set_title(f"{au.replace('_r', '')}: {AU_NAMES.get(au, au)}", fontsize=11)
        ax.set_ylabel('Intensity', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('中性情绪 - 个体内AU分布箱线图', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/boxplot_key_au_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_radar(results, output_dir):
    """创建雷达图"""
    selected_aus = ['AU01_r', 'AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU15_r', 'AU17_r']
    
    angles = np.linspace(0, 2 * np.pi, len(selected_aus), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = {'M1': '#3498db', 'M2': '#2ecc71', 'F1': '#e74c3c'}
    
    for subj in ['M1', 'M2', 'F1']:
        values = [results[subj]['stats'][au]['mean'] for au in selected_aus]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2.5, markersize=8,
               label=results[subj]['info']['name'], color=colors[subj])
        ax.fill(angles, values, alpha=0.15, color=colors[subj])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([au.replace('_r', '') for au in selected_aus], fontsize=11)
    ax.set_title('中性情绪 - 个体AU轮廓雷达图', fontsize=14, pad=30)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/radar_individual_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_time_series(results, output_dir):
    """创建时间序列图"""
    key_aus = ['AU04_r', 'AU07_r', 'AU12_r']
    
    for au in key_aus:
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        
        for idx, subj in enumerate(['M1', 'M2', 'F1']):
            ax = axes[idx]
            data = results[subj]['raw'].get(au, [])
            
            # 采样显示
            if len(data) > 1000:
                sampled = data[::10]
            else:
                sampled = data
            
            x = range(len(sampled))
            color = '#3498db' if results[subj]['info']['gender'] == 'male' else '#e74c3c'
            ax.plot(x, sampled, color=color, alpha=0.7, linewidth=0.8)
            ax.set_ylabel('Intensity', fontsize=10)
            ax.set_title(f"{results[subj]['info']['name']}: {au.replace('_r', '')} ({AU_NAMES.get(au, au)})", 
                        fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(data) * 1.1 if data else 1)
        
        axes[-1].set_xlabel('Time (sampled)', fontsize=11)
        fig.suptitle(f'中性情绪 - {au.replace("_r", "")} 时间序列', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/timeseries_{au.replace("_r", "")}.png', dpi=150, bbox_inches='tight')
        plt.close()

def create_statistics(results, output_dir):
    """生成统计表格"""
    # 1. 保存CSV统计表
    with open(f'{output_dir}/au_statistics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['AU', '被试', '性别', '均值', '标准差', '中位数', '最小值', '最大值', '帧数'])
        for au in KEY_AUS:
            for subj in ['M1', 'M2', 'F1']:
                s = results[subj]['stats'][au]
                info = results[subj]['info']
                writer.writerow([au, info['name'], info['gender'], 
                               f"{s['mean']:.4f}", f"{s['std']:.4f}", 
                               f"{s['median']:.4f}", f"{s['min']:.4f}", 
                               f"{s['max']:.4f}", s['n']])
    
    # 2. 计算性别差异
    with open(f'{output_dir}/gender_comparison_stats.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['AU', '男性1均值', '男性2均值', '男性均值', '女性均值', 
                        '性别差(男-女)', '男性内部差|M1-M2|'])
        for au in KEY_AUS:
            m1 = results['M1']['stats'][au]['mean']
            m2 = results['M2']['stats'][au]['mean']
            f1 = results['F1']['stats'][au]['mean']
            male_avg = (m1 + m2) / 2
            gender_diff = male_avg - f1
            male_internal_diff = abs(m1 - m2)
            writer.writerow([au, f"{m1:.4f}", f"{m2:.4f}", f"{male_avg:.4f}", 
                           f"{f1:.4f}", f"{gender_diff:.4f}", f"{male_internal_diff:.4f}"])
    
    # 3. 创建效应量条形图
    fig, ax = plt.subplots(figsize=(14, 6))
    
    au_short = [au.replace('_r', '') for au in KEY_AUS]
    gender_diffs = []
    male_internal_diffs = []
    
    for au in KEY_AUS:
        m1 = results['M1']['stats'][au]['mean']
        m2 = results['M2']['stats'][au]['mean']
        f1 = results['F1']['stats'][au]['mean']
        gender_diffs.append((m1 + m2)/2 - f1)
        male_internal_diffs.append(abs(m1 - m2))
    
    x = np.arange(len(KEY_AUS))
    width = 0.35
    
    ax.bar(x - width/2, gender_diffs, width, label='性别差 (男-女)', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, male_internal_diffs, width, label='男性内部差 |M1-M2|', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Difference', fontsize=12)
    ax.set_title('中性情绪 - AU差异对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(au_short, fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/statistics_differences.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_report(results, output_dir):
    """生成分析报告"""
    report = []
    report.append("=" * 80)
    report.append("中性情绪个体内差异与性别比较分析报告")
    report.append("日期: 2025-02-17")
    report.append("=" * 80)
    report.append("")
    
    report.append("1. 数据概况")
    report.append("-" * 40)
    report.append("情绪类型: 中性")
    report.append("数据来源: OpenFace 2.0 AU强度值")
    report.append("")
    report.append("样本量:")
    for subj in ['M1', 'M2', 'F1']:
        info = results[subj]['info']
        n = results[subj]['stats'][KEY_AUS[0]]['n']
        report.append(f"  • {info['name']} ({info['gender']}): {n} 帧")
    report.append("")
    
    report.append("2. 核心发现")
    report.append("-" * 40)
    
    # 计算性别差异排名
    gender_diffs = []
    for au in KEY_AUS:
        m1 = results['M1']['stats'][au]['mean']
        m2 = results['M2']['stats'][au]['mean']
        f1 = results['F1']['stats'][au]['mean']
        male_avg = (m1 + m2) / 2
        gender_diffs.append((au, abs(male_avg - f1), male_avg - f1))
    gender_diffs.sort(reverse=True, key=lambda x: x[1])
    
    report.append("最具性别差异的AU (|男-女| 排序):")
    for i, (au, diff, signed_diff) in enumerate(gender_diffs[:5], 1):
        direction = "男性>女性" if signed_diff > 0 else "女性>男性"
        report.append(f"  {i}. {au.replace('_r', '')}: |diff|={diff:.3f} ({direction})")
    
    report.append("")
    
    # 男性内部差异
    male_internal = []
    for au in KEY_AUS:
        m1 = results['M1']['stats'][au]['mean']
        m2 = results['M2']['stats'][au]['mean']
        male_internal.append((au, abs(m1 - m2), m1, m2))
    male_internal.sort(reverse=True, key=lambda x: x[1])
    
    report.append("男性内部个体差异最大的AU (|M1-M2| 排序):")
    for i, (au, diff, m1, m2) in enumerate(male_internal[:5], 1):
        report.append(f"  {i}. {au.replace('_r', '')}: |diff|={diff:.3f} (M1={m1:.3f}, M2={m2:.3f})")
    
    report.append("")
    report.append("3. 被试特异性模式")
    report.append("-" * 40)
    
    for subj in ['M1', 'M2', 'F1']:
        info = results[subj]['info']
        report.append(f"\n{info['name']} ({info['gender']}) 特征:")
        
        # 找出该被试激活最高的3个AU
        au_means = [(au, results[subj]['stats'][au]['mean']) for au in KEY_AUS]
        au_means.sort(reverse=True, key=lambda x: x[1])
        for au, mean in au_means[:3]:
            if mean > 0.1:
                report.append(f"  • {au.replace('_r', '')} ({AU_NAMES.get(au, au)}): {mean:.3f}")
    
    report.append("")
    report.append("4. 输出文件清单")
    report.append("-" * 40)
    report.append("heatmaps/")
    report.append("  - heatmap_M1.png              (男性1 AU激活热力图)")
    report.append("  - heatmap_M2.png              (男性2 AU激活热力图)")
    report.append("  - heatmap_F1.png              (女性1 AU激活热力图)")
    report.append("  - heatmap_3subjects.png       (3被试对比热力图)")
    report.append("  - heatmap_gender_diff.png     (性别差异热力图)")
    report.append("")
    report.append("barplots/")
    report.append("  - barplot_3subjects_comparison.png")
    report.append("  - barplot_gender_comparison.png")
    report.append("")
    report.append("boxplots/")
    report.append("  - boxplot_key_au_distribution.png")
    report.append("")
    report.append("radar/")
    report.append("  - radar_individual_profiles.png")
    report.append("")
    report.append("time_series/")
    report.append("  - timeseries_AU04.png")
    report.append("  - timeseries_AU07.png")
    report.append("  - timeseries_AU12.png")
    report.append("")
    report.append("statistics/")
    report.append("  - au_statistics.csv")
    report.append("  - gender_comparison_stats.csv")
    report.append("  - statistics_differences.png")
    report.append("")
    report.append("raw_data/")
    report.append("  - 中性_M1_male.csv")
    report.append("  - 中性_M2_male.csv")
    report.append("  - 中性_F1_female.csv")
    report.append("")
    report.append("=" * 80)
    
    with open(f'{output_dir}/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)

def main():
    print("=" * 70)
    print("中性情绪个体内差异与性别比较分析")
    print("=" * 70)
    print()
    
    # 创建目录结构
    base_dir = '/root/.openclaw/workspace/analysis_results/2025-02-17_中性情绪_性别对比'
    
    # 清理旧文件
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    os.makedirs(base_dir)
    os.makedirs(f'{base_dir}/heatmaps')
    os.makedirs(f'{base_dir}/barplots')
    os.makedirs(f'{base_dir}/boxplots')
    os.makedirs(f'{base_dir}/radar')
    os.makedirs(f'{base_dir}/time_series')
    os.makedirs(f'{base_dir}/statistics')
    os.makedirs(f'{base_dir}/raw_data')
    
    # 读取数据
    inbound_dir = '/root/.openclaw/media/inbound/'
    results = {}
    
    print("读取中性情绪数据...")
    for filename, info in SUBJECTS.items():
        filepath = os.path.join(inbound_dir, filename)
        data = read_csv(filepath)
        stats = {au: calc_stats(data[au]) for au in KEY_AUS}
        results[info['id']] = {'info': info, 'stats': stats, 'raw': data}
        print(f"  ✓ {info['name']}: {stats[KEY_AUS[0]]['n']} 帧")
    
    print()
    
    # 生成热力图
    print("生成热力图...")
    for subj in ['M1', 'M2', 'F1']:
        name = results[subj]['info']['name']
        create_subject_heatmap(results[subj]['raw'], results[subj]['stats'], 
                              name, f'{base_dir}/heatmaps/heatmap_{subj}.png')
    create_comparison_heatmap(results, f'{base_dir}/heatmaps/heatmap_3subjects.png')
    create_gender_diff_heatmap(results, f'{base_dir}/heatmaps/heatmap_gender_diff.png')
    
    # 生成柱状图
    print("生成柱状图...")
    create_barplots(results, f'{base_dir}/barplots')
    
    # 生成箱线图
    print("生成箱线图...")
    create_boxplots(results, f'{base_dir}/boxplots')
    
    # 生成雷达图
    print("生成雷达图...")
    create_radar(results, f'{base_dir}/radar')
    
    # 生成时间序列
    print("生成时间序列图...")
    create_time_series(results, f'{base_dir}/time_series')
    
    # 生成统计
    print("生成统计数据...")
    create_statistics(results, f'{base_dir}/statistics')
    
    # 复制原始数据
    print("复制原始数据...")
    for filename, info in SUBJECTS.items():
        src = os.path.join(inbound_dir, filename)
        dst = os.path.join(base_dir, f"raw_data/中性_{info['id']}_{info['gender']}.csv")
        shutil.copy2(src, dst)
    
    # 生成报告
    print("生成分析报告...")
    report = generate_report(results, base_dir)
    
    print()
    print("=" * 70)
    print(f"✓ 分析完成！结果保存在: {base_dir}")
    print("=" * 70)
    print()
    print(report[:2000])
    print("...")

if __name__ == '__main__':
    main()
