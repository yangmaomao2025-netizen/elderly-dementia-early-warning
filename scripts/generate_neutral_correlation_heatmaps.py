#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中性情绪AU相关性热力图 - 17×17相关性矩阵
"""

import csv
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 中性情绪数据文件映射
SUBJECTS = {
    'file_18---73cb1d9c-9f3c-4f21-917a-ae9408962385.csv': {'id': 'M1', 'gender': 'male', 'name': '男性1'},
    'file_19---476a6dde-2bc6-48b4-89d3-8c3e70cbd0fd.csv': {'id': 'M2', 'gender': 'male', 'name': '男性2'},
    'file_20---333e020a-bdf5-44a5-b833-c3179c272ccc.csv': {'id': 'F1', 'gender': 'female', 'name': '女性1'}
}

KEY_AUS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
           'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
           'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

def read_csv(filepath):
    """读取CSV并提取AU数据"""
    data = {au: [] for au in KEY_AUS}
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

def create_correlation_heatmap(data, subject_name, output_path):
    """创建17×17 AU相关性热力图"""
    # 构建DataFrame
    df = pd.DataFrame(data)
    
    # 检查并移除常量列（标准差为0的列）
    constant_cols = [col for col in df.columns if df[col].std() == 0]
    if constant_cols:
        print(f"    警告: {subject_name} 的以下AU无变化（设为0相关）: {[c.replace('_r', '') for c in constant_cols]}")
    
    # 计算相关性矩阵
    corr_matrix = df.corr()
    
    # 将NaN替换为0（常量列与其他列无相关性）
    corr_matrix = corr_matrix.fillna(0)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # 使用RdBu_r diverging colormap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # 只显示下三角
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax,
                annot_kws={'size': 8})
    
    # 设置标签
    au_labels = [au.replace('_r', '') for au in KEY_AUS]
    ax.set_xticklabels(au_labels, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(au_labels, rotation=0, fontsize=10)
    
    ax.set_title(f'中性情绪 - {subject_name} AU相关性矩阵 (17×17)', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return corr_matrix

def create_combined_correlation_heatmap(results, output_path):
    """创建3被试AU相关性对比热力图 (合并显示)"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, subj in enumerate(['M1', 'M2', 'F1']):
        ax = axes[idx]
        data = results[subj]['raw']
        df = pd.DataFrame(data)
        corr_matrix = df.corr().fillna(0)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=False,  # 合并图不显示数值，避免拥挤
                    cmap='RdBu_r',
                    vmin=-1, vmax=1,
                    center=0,
                    square=True,
                    linewidths=0.3,
                    cbar_kws={"shrink": 0.8},
                    ax=ax)
        
        au_labels = [au.replace('_r', '') for au in KEY_AUS]
        ax.set_xticklabels(au_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(au_labels if idx == 0 else [], rotation=0, fontsize=8)
        ax.set_title(f"{results[subj]['info']['name']}", fontsize=12)
    
    fig.suptitle('中性情绪 - 3被试AU相关性对比', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_gender_average_correlation(results, output_path):
    """创建性别平均相关性热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # 男性平均相关性 (M1 + M2) / 2
    male_data = {}
    for au in KEY_AUS:
        m1_data = results['M1']['raw'][au]
        m2_data = results['M2']['raw'][au]
        # 对齐长度取平均
        min_len = min(len(m1_data), len(m2_data))
        male_data[au] = [(m1_data[i] + m2_data[i]) / 2 for i in range(min_len)]
    
    df_male = pd.DataFrame(male_data)
    corr_male = df_male.corr().fillna(0)
    
    # 女性相关性
    df_female = pd.DataFrame(results['F1']['raw'])
    corr_female = df_female.corr().fillna(0)
    
    # 绘制男性
    mask = np.triu(np.ones_like(corr_male, dtype=bool), k=1)
    sns.heatmap(corr_male, 
                mask=mask,
                annot=False,
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                center=0,
                square=True,
                linewidths=0.3,
                cbar_kws={"shrink": 0.8},
                ax=axes[0])
    au_labels = [au.replace('_r', '') for au in KEY_AUS]
    axes[0].set_xticklabels(au_labels, rotation=45, ha='right', fontsize=9)
    axes[0].set_yticklabels(au_labels, rotation=0, fontsize=9)
    axes[0].set_title('男性平均 (M1+M2)/2', fontsize=12)
    
    # 绘制女性
    sns.heatmap(corr_female, 
                mask=mask,
                annot=False,
                cmap='RdBu_r',
                vmin=-1, vmax=1,
                center=0,
                square=True,
                linewidths=0.3,
                cbar_kws={"shrink": 0.8},
                ax=axes[1])
    axes[1].set_xticklabels(au_labels, rotation=45, ha='right', fontsize=9)
    axes[1].set_yticklabels([])
    axes[1].set_title('女性 (F1)', fontsize=12)
    
    fig.suptitle('中性情绪 - 性别AU相关性对比', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=" * 70)
    print("中性情绪AU相关性热力图分析")
    print("=" * 70)
    print()
    
    output_dir = '/root/.openclaw/workspace/analysis_results/2025-02-17_中性情绪_性别对比/heatmaps'
    os.makedirs(output_dir, exist_ok=True)
    
    inbound_dir = '/root/.openclaw/media/inbound/'
    results = {}
    
    print("读取数据...")
    for filename, info in SUBJECTS.items():
        filepath = os.path.join(inbound_dir, filename)
        data = read_csv(filepath)
        results[info['id']] = {'info': info, 'raw': data}
        print(f"  ✓ {info['name']}: {len(data[KEY_AUS[0]])} 帧")
    
    print()
    print("生成AU相关性热力图...")
    
    # 生成单个被试的相关性热力图
    for subj in ['M1', 'M2', 'F1']:
        name = results[subj]['info']['name']
        output_path = f'{output_dir}/correlation_matrix_{subj}.png'
        corr = create_correlation_heatmap(results[subj]['raw'], name, output_path)
        print(f"  ✓ {name} 相关性矩阵: {output_path}")
    
    # 生成3被试对比热力图
    print("  生成3被试对比图...")
    create_combined_correlation_heatmap(results, f'{output_dir}/correlation_matrix_3subjects.png')
    
    # 生成性别平均相关性热力图
    print("  生成性别对比图...")
    create_gender_average_correlation(results, f'{output_dir}/correlation_matrix_gender.png')
    
    print()
    print("=" * 70)
    print(f"✓ 相关性分析完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
