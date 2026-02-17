#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成女性患者AU相关矩阵热图（17×17）
处理NaN值（常数列填充0）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 文件映射
file_mapping = {
    'F1': {
        '悲伤': '/root/.openclaw/media/inbound/file_39---a156b700-0dc5-4a7d-98a7-842a4d9ee1da.csv',
        '中性': '/root/.openclaw/media/inbound/file_33---0952895f-e350-4674-8b75-1c310d208392.csv',
        '积极': '/root/.openclaw/media/inbound/file_36---182b42a2-87e5-41cf-af26-d505d000e246.csv'
    },
    'F2': {
        '悲伤': '/root/.openclaw/media/inbound/file_40---571ec375-dfd5-403f-b0da-0922d9ff79e6.csv',
        '中性': '/root/.openclaw/media/inbound/file_34---02e0b344-096c-4493-99ba-38f427cc085a.csv',
        '积极': '/root/.openclaw/media/inbound/file_37---c15a0c41-ee72-464a-b765-40030db85592.csv'
    },
    'F3': {
        '悲伤': '/root/.openclaw/media/inbound/file_41---90b39ad4-e124-4e19-84ad-d2beb4ddcc4e.csv',
        '中性': '/root/.openclaw/media/inbound/file_35---1d02ab72-301b-4468-8579-ed6bf3c7152a.csv',
        '积极': '/root/.openclaw/media/inbound/file_38---0e31b32a-8a78-4f18-9127-33ab67cbf863.csv'
    }
}

au_columns = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def generate_correlation_heatmap(data, title, output_path):
    """生成17×17 AU相关矩阵热图"""
    # 计算相关系数
    corr_matrix = data[au_columns].corr()
    
    # 处理NaN（常数列）
    nan_count = corr_matrix.isna().sum().sum()
    if nan_count > 0:
        print(f"  警告: {nan_count} 个NaN值（常数AU），填充为0")
        corr_matrix = corr_matrix.fillna(0)
    
    # 绘制热图
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
    
    ax.set_title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  保存: {output_path.name}")
    return corr_matrix

def main():
    output_dir = Path('/root/.openclaw/workspace/analysis_results/2026-02-17_女性患者跨情绪分析/heatmaps')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("生成女性患者AU相关矩阵热图（17×17）")
    print("="*70)
    
    for patient_id, emotions in file_mapping.items():
        print(f"\n{patient_id}:")
        for emotion, filepath in emotions.items():
            df = load_data(filepath)
            frame_count = len(df)
            
            title_suffix = " ⚠️数据不完整" if patient_id == 'F1' and emotion == '中性' else ""
            title = f'{patient_id} - {emotion}情绪\n({frame_count} frames){title_suffix}'
            output_path = output_dir / f'{patient_id}_{emotion}_correlation_matrix.png'
            
            generate_correlation_heatmap(df, title, output_path)
    
    print(f"\n{'='*70}")
    print("相关矩阵热图生成完成！")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
