#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
悲伤情绪个体内差异与性别比较分析 - 修复版
"""

import csv
import os
import math
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SUBJECTS = {
    'file_15---c598d66b-d56c-4419-b31c-5d06bb412970.csv': {'id': 'M1', 'gender': 'male', 'name': '男性1'},
    'file_16---6d147c2c-4114-4a63-a1d3-ca8e6c8c76e2.csv': {'id': 'M2', 'gender': 'male', 'name': '男性2'},
    'file_17---177cc846-8ba4-4e5b-918b-f1e2d3588325.csv': {'id': 'F1', 'gender': 'female', 'name': '女性1'}
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
    """读取CSV并提取AU数据 - 处理带空格的列名"""
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
        return {'mean': 0, 'std': 0, 'n': 0}
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0
    std = math.sqrt(variance)
    return {'mean': mean, 'std': std, 'n': n}

def main():
    print("=" * 70)
    print("悲伤情绪个体内差异与性别比较分析")
    print("=" * 70)
    print()
    
    output_dir = '/root/.openclaw/workspace/analysis_results/2025-02-17_悲伤情绪_性别对比'
    os.makedirs(output_dir, exist_ok=True)
    
    inbound_dir = '/root/.openclaw/media/inbound/'
    results = {}
    
    # 读取数据
    for filename, info in SUBJECTS.items():
        filepath = os.path.join(inbound_dir, filename)
        data = read_csv(filepath)
        stats = {au: calc_stats(data[au]) for au in KEY_AUS}
        results[info['id']] = {'info': info, 'stats': stats, 'raw': data}
        print(f"✓ {info['name']}: {stats[KEY_AUS[0]]['n']} 帧")
    
    print()
    
    # 创建热力图
    print("生成: 热力图...")
    fig, ax = plt.subplots(figsize=(8, 14))
    matrix = [[results[s]['stats'][au]['mean'] for s in ['M1', 'M2', 'F1']] for au in KEY_AUS]
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(['男性1', '男性2', '女性1'], fontsize=12)
    ax.set_yticks(range(len(KEY_AUS)))
    ax.set_yticklabels([f"{au}\n({AU_NAMES.get(au, au)})" for au in KEY_AUS], fontsize=9)
    for i in range(len(KEY_AUS)):
        for j in range(3):
            ax.text(j, i, f'{matrix[i][j]:.2f}', ha="center", va="center", color="black", fontsize=8)
    ax.set_title('悲伤情绪 - 3被试AU均值热力图', fontsize=14, pad=20)
    plt.colorbar(im, ax=ax, label='AU Intensity (Mean)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/heatmap_3subjects.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 性别对比柱状图
    print("生成: 性别对比图...")
    male_aus = defaultdict(list)
    female_aus = defaultdict(list)
    for subj_id, data in results.items():
        gender = data['info']['gender']
        for au in KEY_AUS:
            if au in data['raw'] and data['raw'][au]:
                if gender == 'male':
                    male_aus[au].extend(data['raw'][au])
                else:
                    female_aus[au].extend(data['raw'][au])
    
    male_means = [sum(male_aus.get(au, [0]))/len(male_aus.get(au, [1])) if male_aus.get(au) else 0 for au in KEY_AUS]
    female_means = [sum(female_aus.get(au, [0]))/len(female_aus.get(au, [1])) if female_aus.get(au) else 0 for au in KEY_AUS]
    
    fig, ax = plt.subplots(figsize=(16, 7))
    x = list(range(len(KEY_AUS)))
    width = 0.35
    ax.bar([i - width/2 for i in x], male_means, width, label='男性 (n=2)', color='#3498db', alpha=0.8)
    ax.bar([i + width/2 for i in x], female_means, width, label='女性 (n=1)', color='#e74c3c', alpha=0.8)
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title('悲伤情绪性别差异: AU均值对比', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{au}\n{AU_NAMES.get(au, '')}" for au in KEY_AUS], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/gender_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 个体折线图
    print("生成: 个体轮廓图...")
    selected_aus = ['AU01_r', 'AU04_r', 'AU06_r', 'AU12_r', 'AU15_r', 'AU17_r', 'AU20_r']
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {'M1': '#3498db', 'M2': '#2ecc71', 'F1': '#e74c3c'}
    markers = {'M1': 'o', 'M2': 's', 'F1': '^'}
    x = list(range(len(selected_aus)))
    for subj_id in ['M1', 'M2', 'F1']:
        values = [results[subj_id]['stats'].get(au, {}).get('mean', 0) for au in selected_aus]
        ax.plot(x, values, marker=markers[subj_id], linewidth=2.5, markersize=8, 
               label=results[subj_id]['info']['name'], color=colors[subj_id], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{au}\n({AU_NAMES.get(au, au)})" for au in selected_aus], fontsize=10)
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title('悲伤情绪AU轮廓 - 个体对比', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/individual_profiles.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 关键AU箱线图
    print("生成: 变异性分析图...")
    key_aus_plot = ['AU01_r', 'AU04_r', 'AU06_r', 'AU12_r', 'AU15_r', 'AU17_r']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, au in enumerate(key_aus_plot):
        ax = axes[idx]
        box_data = []
        labels = []
        for subj_id in ['M1', 'M2', 'F1']:
            if au in results[subj_id]['raw'] and results[subj_id]['raw'][au]:
                box_data.append(results[subj_id]['raw'][au])
                labels.append(results[subj_id]['info']['name'])
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            for patch, subj_id in zip(bp['boxes'], ['M1', 'M2', 'F1']):
                color = '#3498db' if results[subj_id]['info']['gender'] == 'male' else '#e74c3c'
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        ax.set_title(f"{au}: {AU_NAMES.get(au, au)}", fontsize=11)
        ax.set_ylabel('Intensity', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('悲伤情绪个体内AU变异性分析', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/variance_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 个体间差异热力图
    print("生成: 差异热力图...")
    subjects = ['M1', 'M2', 'F1']
    diff_matrix = []
    comparisons = []
    for i, s1 in enumerate(subjects):
        for j, s2 in enumerate(subjects):
            if i < j:
                comparisons.append(f"{results[s1]['info']['name']} vs {results[s2]['info']['name']}")
                row = [abs(results[s1]['stats'][au]['mean'] - results[s2]['stats'][au]['mean']) for au in KEY_AUS]
                diff_matrix.append(row)
    if diff_matrix:
        fig, ax = plt.subplots(figsize=(14, 4))
        cax = ax.imshow(diff_matrix, cmap='Reds', aspect='auto')
        ax.set_xticks(range(len(KEY_AUS)))
        ax.set_xticklabels([au.replace('_r', '') for au in KEY_AUS], rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(comparisons)))
        ax.set_yticklabels(comparisons, fontsize=10)
        for i in range(len(comparisons)):
            for j in range(len(KEY_AUS)):
                text = ax.text(j, i, f'{diff_matrix[i][j]:.2f}', ha="center", va="center", 
                              color="white" if diff_matrix[i][j] > 0.5 else "black", fontsize=8)
        ax.set_title('个体间AU差异绝对值', fontsize=12, pad=15)
        plt.colorbar(cax, ax=ax, label='|Difference|')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/differences_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print()
    print("=" * 70)
    print(f"✓ 所有图表生成完成！保存位置: {output_dir}")
    print("=" * 70)
    
    # 打印关键发现
    print()
    print("【关键发现摘要】")
    diffs = []
    for au in KEY_AUS:
        m1 = results['M1']['stats'][au]['mean']
        m2 = results['M2']['stats'][au]['mean']
        f1 = results['F1']['stats'][au]['mean']
        male_avg = (m1 + m2) / 2
        diffs.append((au, abs(f1 - male_avg), f1 - male_avg))
    diffs.sort(reverse=True, key=lambda x: x[1])
    print("\n1. 性别差异最大的AU (Top 5):")
    for au, abs_diff, diff in diffs[:5]:
        direction = "女性>男性" if diff > 0 else "男性>女性"
        print(f"   - {au} ({AU_NAMES.get(au, '')}): |diff|={abs_diff:.3f} ({direction})")

if __name__ == '__main__':
    main()
