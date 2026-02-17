#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
悲伤情绪个体内差异与性别比较分析
分析3个被试（2男1女）的悲伤情绪AU数据
"""

import csv
import os
import json
import statistics
import math
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # 无GUI模式
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义被试信息
SUBJECTS = {
    'file_15---c598d66b-d56c-4419-b31c-5d06bb412970.csv': {'id': 'M1', 'gender': 'male', 'name': '男性1'},
    'file_16---6d147c2c-4114-4a63-a1d3-ca8e6c8c76e2.csv': {'id': 'M2', 'gender': 'male', 'name': '男性2'},
    'file_17---177cc846-8ba4-4e5b-918b-f1e2d3588325.csv': {'id': 'F1', 'gender': 'female', 'name': '女性1'}
}

# 关键AU定义
KEY_AUS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
           'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
           'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

AU_NAMES = {
    'AU01_r': '内侧眉毛上扬',
    'AU02_r': '外侧眉毛上扬', 
    'AU04_r': '眉毛下垂',
    'AU05_r': '上眼睑上扬',
    'AU06_r': '脸颊上扬',
    'AU07_r': '眼睑收紧',
    'AU09_r': '鼻子起皱',
    'AU10_r': '上唇上扬',
    'AU12_r': '嘴角上扬',
    'AU14_r': '嘴角收紧',
    'AU15_r': '嘴角下垂',
    'AU17_r': '下巴上扬',
    'AU20_r': '嘴唇伸展',
    'AU23_r': '嘴唇收紧',
    'AU25_r': '嘴唇分开',
    'AU26_r': '下巴下降',
    'AU45_r': '眨眼'
}

def read_csv_file(filepath):
    """读取CSV文件并提取AU数据"""
    data = defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for au in KEY_AUS:
                if au in row:
                    try:
                        val = float(row[au])
                        data[au].append(val)
                    except:
                        pass
    return data

def calculate_stats(values):
    """计算统计数据"""
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'n': 0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0
    std = math.sqrt(variance)
    sorted_vals = sorted(values)
    median = sorted_vals[n // 2] if n % 2 == 1 else (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    return {
        'mean': mean,
        'std': std,
        'min': min(values),
        'max': max(values),
        'median': median,
        'n': n
    }

def analyze_subjects():
    """分析所有被试数据"""
    inbound_dir = '/root/.openclaw/media/inbound/'
    results = {}
    
    for filename, info in SUBJECTS.items():
        filepath = os.path.join(inbound_dir, filename)
        if os.path.exists(filepath):
            print(f"分析: {info['name']} ({filename[:20]}...)")
            data = read_csv_file(filepath)
            
            stats_data = {}
            for au in KEY_AUS:
                if au in data and data[au]:
                    stats_data[au] = calculate_stats(data[au])
            
            results[info['id']] = {
                'info': info,
                'raw_data': data,
                'stats': stats_data
            }
    
    return results

def create_heatmap(results, output_dir):
    """创建热力图 - 3个被试的AU均值对比"""
    subjects = ['M1', 'M2', 'F1']
    
    matrix = []
    for au in KEY_AUS:
        row = []
        for subj in subjects:
            if subj in results and au in results[subj]['stats']:
                row.append(results[subj]['stats'][au]['mean'])
            else:
                row.append(0)
        matrix.append(row)
    
    fig, ax = plt.subplots(figsize=(8, 14))
    
    # 使用matshow或imshow
    cax = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    
    # 设置标签
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels([results[s]['info']['name'] for s in subjects], fontsize=12)
    
    au_labels = [f"{au}\n({AU_NAMES.get(au, au)})" for au in KEY_AUS]
    ax.set_yticks(range(len(KEY_AUS)))
    ax.set_yticklabels(au_labels, fontsize=9)
    
    # 添加数值
    for i in range(len(KEY_AUS)):
        for j in range(len(subjects)):
            text = ax.text(j, i, f'{matrix[i][j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('悲伤情绪 - 3被试AU均值热力图\n(Sadness: Individual AU Comparison)', fontsize=14, pad=20)
    plt.colorbar(cax, ax=ax, label='AU Intensity (Mean)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_3subjects_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: heatmap_3subjects_comparison.png")

def create_gender_comparison(results, output_dir):
    """创建性别对比图"""
    male_aus = defaultdict(list)
    female_aus = defaultdict(list)
    
    for subj_id, data in results.items():
        gender = data['info']['gender']
        for au in KEY_AUS:
            if au in data['raw_data']:
                if gender == 'male':
                    male_aus[au].extend(data['raw_data'][au])
                else:
                    female_aus[au].extend(data['raw_data'][au])
    
    male_means = []
    female_means = []
    for au in KEY_AUS:
        m_vals = male_aus.get(au, [0])
        f_vals = female_aus.get(au, [0])
        male_means.append(sum(m_vals) / len(m_vals) if m_vals else 0)
        female_means.append(sum(f_vals) / len(f_vals) if f_vals else 0)
    
    fig, ax = plt.subplots(figsize=(16, 7))
    
    x = list(range(len(KEY_AUS)))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], male_means, width, label='男性 (n=2)', color='#3498db', alpha=0.8)
    bars2 = ax.bar([i + width/2 for i in x], female_means, width, label='女性 (n=1)', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title('悲伤情绪性别差异: AU均值对比\n(Gender Comparison in Sadness)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{au}\n{AU_NAMES.get(au, '')}" for au in KEY_AUS], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_comparison_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: gender_comparison_bar.png")

def create_individual_lines(results, output_dir):
    """创建个体折线图"""
    selected_aus = ['AU01_r', 'AU04_r', 'AU06_r', 'AU12_r', 'AU15_r', 'AU17_r', 'AU20_r']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'M1': '#3498db', 'M2': '#2ecc71', 'F1': '#e74c3c'}
    markers = {'M1': 'o', 'M2': 's', 'F1': '^'}
    
    x = list(range(len(selected_aus)))
    
    for subj_id in ['M1', 'M2', 'F1']:
        if subj_id in results:
            values = []
            for au in selected_aus:
                if au in results[subj_id]['stats']:
                    values.append(results[subj_id]['stats'][au]['mean'])
                else:
                    values.append(0)
            
            label = results[subj_id]['info']['name']
            ax.plot(x, values, marker=markers[subj_id], linewidth=2.5, markersize=8, 
                   label=label, color=colors[subj_id], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"{au}\n({AU_NAMES.get(au, au)})" for au in selected_aus], fontsize=10)
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title('悲伤情绪AU轮廓 - 个体对比\n(Individual AU Profiles)', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'individual_line_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: individual_line_profiles.png")

def create_variance_analysis(results, output_dir):
    """创建个体内变异性分析"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    key_aus_plot = ['AU01_r', 'AU04_r', 'AU06_r', 'AU12_r', 'AU15_r', 'AU17_r']
    
    for idx, au in enumerate(key_aus_plot):
        ax = axes[idx]
        
        box_data = []
        labels = []
        colors = []
        
        for subj_id in ['M1', 'M2', 'F1']:
            if subj_id in results and au in results[subj_id]['raw_data'] and results[subj_id]['raw_data'][au]:
                box_data.append(results[subj_id]['raw_data'][au])
                labels.append(results[subj_id]['info']['name'])
                colors.append('#3498db' if results[subj_id]['info']['gender'] == 'male' else '#e74c3c')
        
        if box_data and labels:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax.set_title(f"{au}: {AU_NAMES.get(au, au)}", fontsize=11)
        ax.set_ylabel('Intensity', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    fig.suptitle('悲伤情绪个体内AU变异性分析\n(Intra-individual AU Variability)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'individual_variance_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: individual_variance_boxplot.png")

def create_difference_heatmap(results, output_dir):
    """创建个体差异热力图"""
    subjects = ['M1', 'M2', 'F1']
    
    # 计算两两差异
    diff_matrix = []
    comparisons = []
    
    for i, s1 in enumerate(subjects):
        for j, s2 in enumerate(subjects):
            if i < j:
                comparisons.append(f"{results[s1]['info']['name']} vs {results[s2]['info']['name']}")
                row = []
                for au in KEY_AUS:
                    v1 = results[s1]['stats'].get(au, {}).get('mean', 0) if s1 in results else 0
                    v2 = results[s2]['stats'].get(au, {}).get('mean', 0) if s2 in results else 0
                    row.append(abs(v1 - v2))
                diff_matrix.append(row)
    
    if diff_matrix:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        cax = ax.imshow(diff_matrix, cmap='Reds', aspect='auto')
        
        ax.set_xticks(range(len(KEY_AUS)))
        ax.set_xticklabels([au.replace('_r', '') for au in KEY_AUS], rotation=45, ha='right', fontsize=9)
        
        ax.set_yticks(range(len(comparisons)))
        ax.set_yticklabels(comparisons, fontsize=10)
        
        for i in range(len(comparisons)):
            for j in range(len(KEY_AUS)):
                text = ax.text(j, i, f'{diff_matrix[i][j]:.2f}',
                              ha="center", va="center", color="white" if diff_matrix[i][j] > 0.5 else "black", fontsize=8)
        
        ax.set_title('个体间AU差异绝对值\n(Inter-individual AU Differences)', fontsize=12, pad=15)
        plt.colorbar(cax, ax=ax, label='|Difference|')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'individual_differences_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存: individual_differences_heatmap.png")

def generate_report(results, output_dir):
    """生成统计报告"""
    report = []
    report.append("=" * 70)
    report.append("悲伤情绪个体内差异与性别比较分析报告")
    report.append("Sadness Emotion: Individual & Gender Comparison Analysis")
    report.append("=" * 70)
    report.append("")
    
    # 被试信息
    report.append("【被试信息】")
    for subj_id in ['M1', 'M2', 'F1']:
        if subj_id in results:
            info = results[subj_id]['info']
            stats = results[subj_id]['stats']
            total_frames = list(stats.values())[0]['n'] if stats else 0
            report.append(f"  {info['name']}: 性别={info['gender']}, 帧数={total_frames}")
    report.append("")
    
    # AU均值表
    report.append("【AU均值对比表】")
    report.append(f"{'AU':<12} {'中文名':<14} {'男性1':<10} {'男性2':<10} {'女性1':<10} {'男均值':<10} {'性别差':<10}")
    report.append("-" * 80)
    
    for au in KEY_AUS:
        m1 = results['M1']['stats'].get(au, {}).get('mean', 0) if 'M1' in results else 0
        m2 = results['M2']['stats'].get(au, {}).get('mean', 0) if 'M2' in results else 0
        f1 = results['F1']['stats'].get(au, {}).get('mean', 0) if 'F1' in results else 0
        
        male_avg = (m1 + m2) / 2
        diff = f1 - male_avg
        
        au_short = au.replace('_r', '')
        cn_name = AU_NAMES.get(au, '')[:12]
        report.append(f"{au_short:<12} {cn_name:<14} {m1:<10.3f} {m2:<10.3f} {f1:<10.3f} {male_avg:<10.3f} {diff:<10.3f}")
    
    report.append("")
    report.append("【关键发现】")
    
    # 差异最大的AU
    diffs = []
    for au in KEY_AUS:
        m1 = results['M1']['stats'].get(au, {}).get('mean', 0) if 'M1' in results else 0
        m2 = results['M2']['stats'].get(au, {}).get('mean', 0) if 'M2' in results else 0
        f1 = results['F1']['stats'].get(au, {}).get('mean', 0) if 'F1' in results else 0
        male_avg = (m1 + m2) / 2
        diffs.append((au, abs(f1 - male_avg), f1 - male_avg))
    
    diffs.sort(reverse=True, key=lambda x: x[1])
    
    report.append(f"  1. 性别差异最大的AU (Top 5):")
    for au, abs_diff, diff in diffs[:5]:
        direction = "女性>男性" if diff > 0 else "男性>女性"
        report.append(f"     - {au} ({AU_NAMES.get(au, '')}): |diff|={abs_diff:.3f} ({direction})")
    
    report.append("")
    report.append("  2. 个体内AU变异性 (标准差均值):")
    for subj_id in ['M1', 'M2', 'F1']:
        if subj_id in results:
            info = results[subj_id]['info']
            stats = results[subj_id]['stats']
            if stats:
                avg_std = sum(s['std'] for s in stats.values()) / len(stats)
                report.append(f"     - {info['name']}: 平均AU标准差 = {avg_std:.3f}")
    
    # 男性内部差异
    report.append("")
    report.append("  3. 男性内部差异 (M1 vs M2):")
    for au in KEY_AUS:
        m1 = results['M1']['stats'].get(au, {}).get('mean', 0) if 'M1' in results else 0
        m2 = results['M2']['stats'].get(au, {}).get('mean', 0) if 'M2' in results else 0
        diff = abs(m1 - m2)
        if diff > 0.3:
            report.append(f"     - {au} ({AU_NAMES.get(au, '')}): |M1-M2|={diff:.3f}")
    
    report.append("")
    report.append("=" * 70)
    report.append("分析说明:")
    report.append("  - 本分析基于OpenFace 2.0提取的17个AU强度值")
    report.append("  - 数值为AU强度均值，范围0-5 (AU45为0-1)")
    report.append("  - 性别差异 = 女性均值 - 男性均值 (n=2)")
    report.append("  - 样本量较小(n=3)，结果仅供参考")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"  保存: analysis_report.txt")
    return report_text

def main():
    print("=" * 70)
    print("悲伤情绪个体内差异与性别比较分析")
    print("=" * 70)
    print()
    
    # 创建输出目录
    output_dir = '/root/.openclaw/workspace/analysis_results/2025-02-17_悲伤情绪_性别对比'
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制原始数据
    raw_dir = os.path.join(output_dir, 'raw_data')
    os.makedirs(raw_dir, exist_ok=True)
    
    inbound_dir = '/root/.openclaw/media/inbound/'
    for filename, info in SUBJECTS.items():
        src = os.path.join(inbound_dir, filename)
        if os.path.exists(src):
            dst = os.path.join(raw_dir, f"悲伤_{info['id']}_{info['gender']}.csv")
            os.system(f'cp "{src}" "{dst}"')
    print(f"✓ 原始数据已保存到: {raw_dir}")
    print()
    
    # 分析数据
    print("正在分析数据...")
    results = analyze_subjects()
    print(f"✓ 成功分析 {len(results)} 个被试")
    print()
    
    # 生成可视化
    print("生成可视化图表...")
    create_heatmap(results, output_dir)
    create_gender_comparison(results, output_dir)
    create_individual_lines(results, output_dir)
    create_variance_analysis(results, output_dir)
    create_difference_heatmap(results, output_dir)
    print()
    
    # 生成报告
    print("生成统计报告...")
    report = generate_report(results, output_dir)
    print()
    
    print("=" * 70)
    print(f"✓ 分析完成！结果保存在: {output_dir}")
    print("=" * 70)
    print()
    print("生成的文件:")
    print("  📊 heatmap_3subjects_comparison.png (3被试AU均值热力图)")
    print("  📊 gender_comparison_bar.png (性别对比柱状图)")
    print("  📊 individual_line_profiles.png (个体AU轮廓折线图)")
    print("  📊 individual_variance_boxplot.png (个体内变异性箱线图)")
    print("  📊 individual_differences_heatmap.png (个体间差异热力图)")
    print("  📄 analysis_report.txt (详细统计报告)")
    print()
    
    # 打印报告摘要
    print("-" * 70)
    print(report[:3000])
    print("-" * 70)

if __name__ == '__main__':
    main()
