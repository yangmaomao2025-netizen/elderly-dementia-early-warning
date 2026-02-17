#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成AU相关性矩阵热力图 - 参照上次标准
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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

def calculate_correlation(data):
    """计算AU间的相关系数矩阵"""
    n = len(KEY_AUS)
    corr_matrix = np.zeros((n, n))
    
    for i, au1 in enumerate(KEY_AUS):
        for j, au2 in enumerate(KEY_AUS):
            if au1 in data and au2 in data and len(data[au1]) == len(data[au2]):
                x = np.array(data[au1])
                y = np.array(data[au2])
                # 计算皮尔逊相关系数
                if np.std(x) > 0 and np.std(y) > 0:
                    corr = np.corrcoef(x, y)[0, 1]
                    corr_matrix[i, j] = corr if not np.isnan(corr) else 0
                else:
                    corr_matrix[i, j] = 0
            else:
                corr_matrix[i, j] = 0
    
    return corr_matrix

def create_correlation_heatmap(data, title, output_path):
    """创建相关性热力图 - 参照上次标准"""
    corr_matrix = calculate_correlation(data)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 使用seaborn绘制热力图
    au_labels = [au.replace('_r', '') for au in KEY_AUS]
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                xticklabels=au_labels,
                yticklabels=au_labels,
                cbar_kws={'label': 'Correlation'},
                ax=ax,
                annot_kws={'size': 8})
    
    ax.set_title(f'{title} - AU相关性矩阵', fontsize=14, pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return corr_matrix

def create_all_subjects_comparison():
    """创建3被试对比图（并排显示）"""
    inbound_dir = '/root/.openclaw/media/inbound/'
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    for idx, (filename, info) in enumerate(SUBJECTS.items()):
        filepath = os.path.join(inbound_dir, filename)
        data = read_csv(filepath)
        corr_matrix = calculate_correlation(data)
        
        au_labels = [au.replace('_r', '') for au in KEY_AUS]
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    fmt='.2f',
                    cmap='RdBu_r',
                    center=0,
                    vmin=-1, vmax=1,
                    square=True,
                    xticklabels=au_labels,
                    yticklabels=au_labels if idx == 0 else [],
                    cbar_kws={'label': 'Correlation'},
                    ax=axes[idx],
                    annot_kws={'size': 7})
        
        axes[idx].set_title(f'{info["name"]} - AU相关性', fontsize=13, pad=10)
        axes[idx].tick_params(axis='x', rotation=45, labelsize=9)
        if idx == 0:
            axes[idx].tick_params(axis='y', rotation=0, labelsize=9)
    
    plt.suptitle('悲伤情绪 - 3被试AU相关性矩阵对比', fontsize=16, y=1.02)
    plt.tight_layout()
    
    output_dir = '/root/.openclaw/workspace/analysis_results/2025-02-17_悲伤情绪_性别对比/heatmaps'
    plt.savefig(f'{output_dir}/heatmap_correlation_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ heatmap_correlation_all.png")

def main():
    print("=" * 70)
    print("生成AU相关性矩阵热力图")
    print("=" * 70)
    print()
    
    inbound_dir = '/root/.openclaw/media/inbound/'
    output_dir = '/root/.openclaw/workspace/analysis_results/2025-02-17_悲伤情绪_性别对比/heatmaps'
    
    # 为每个被试生成相关性热力图
    for filename, info in SUBJECTS.items():
        filepath = os.path.join(inbound_dir, filename)
        data = read_csv(filepath)
        
        output_path = f'{output_dir}/heatmap_correlation_{info["id"]}.png'
        create_correlation_heatmap(data, info['name'], output_path)
        print(f"  ✓ heatmap_correlation_{info['id']}.png ({info['name']})")
    
    # 生成3被试对比图
    print()
    print("生成3被试对比图...")
    create_all_subjects_comparison()
    
    print()
    print("=" * 70)
    print("AU相关性热力图生成完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
