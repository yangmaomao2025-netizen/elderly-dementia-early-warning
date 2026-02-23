#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成17×17 AU相关性热图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

def create_au_correlation_heatmap(csv_file, output_path):
    """为单个CSV文件生成AU相关性热图"""
    
    # 读取数据
    df = pd.read_csv(csv_file, on_bad_lines='skip')
    
    # 获取AU_r列
    au_cols = [col for col in df.columns if col.endswith('_r')]
    
    if len(au_cols) == 0:
        print(f"  跳过: {csv_file} (无AU数据)")
        return
    
    # 清理列名
    au_data = df[au_cols].copy()
    au_data.columns = [col.strip().replace('_r', '') for col in au_cols]
    
    # 计算相关性矩阵
    corr_matrix = au_data.corr()
    
    # 创建热图
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # 只显示下三角
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f', 
                cmap='RdBu_r', 
                center=0,
                vmin=-1, vmax=1,
                square=True,
                ax=ax,
                cbar_kws={'label': 'Correlation'},
                annot_kws={'size': 8})
    
    # 获取文件名用于标题
    file_name = os.path.basename(csv_file).replace('.csv', '')
    ax.set_title(f'{file_name} - AU Correlation Matrix (17×17)', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {output_path}")

# 为所有分析文件夹生成相关性热图
def process_all_folders():
    """处理所有分析文件夹"""
    
    base_dir = "/root/.openclaw/workspace/老年失智人群预警模式科研项目/analysis_results"
    
    # 查找所有包含raw_data的文件夹
    all_folders = glob.glob(os.path.join(base_dir, "2025-02-23_*"))
    
    print(f"找到 {len(all_folders)} 个分析文件夹")
    print("="*60)
    
    for folder in sorted(all_folders):
        folder_name = os.path.basename(folder)
        raw_data_dir = os.path.join(folder, "raw_data")
        
        if not os.path.exists(raw_data_dir):
            continue
        
        print(f"\n处理: {folder_name}")
        
        # 查找所有CSV文件
        csv_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))
        
        for csv_file in csv_files:
            # 创建热图目录
            heatmap_dir = os.path.join(folder, "heatmaps")
            os.makedirs(heatmap_dir, exist_ok=True)
            
            # 生成输出文件名
            base_name = os.path.basename(csv_file).replace('.csv', '')
            output_path = os.path.join(heatmap_dir, f"correlation_{base_name}.png")
            
            create_au_correlation_heatmap(csv_file, output_path)

if __name__ == '__main__':
    print("="*60)
    print("生成17×17 AU相关性热图")
    print("="*60)
    process_all_folders()
    print("\n" + "="*60)
    print("✓ 全部完成!")
    print("="*60)
