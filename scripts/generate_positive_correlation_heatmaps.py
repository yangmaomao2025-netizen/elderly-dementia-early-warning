#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
积极情绪AU相关性热力图生成脚本
生成17×17 AU相关性矩阵热力图
处理常数列（零方差AU）- 填充NaN为0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# ============ 配置 ============
FILE_MAPPING = {
    '/root/.openclaw/media/inbound/file_21---c1ecbaad-5700-42b7-a743-1b75f81b7ff1.csv': ('M1', 'Male'),
    '/root/.openclaw/media/inbound/file_22---772490a5-e791-43b9-8f4a-25c2f614570a.csv': ('M2', 'Male'),
    '/root/.openclaw/media/inbound/file_23---06535c58-c474-473b-a68d-aadcee3e3ca7.csv': ('F1', 'Female'),
}

AU_COLUMNS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

AU_SHORT_NAMES = [au.replace('_r', '') for au in AU_COLUMNS]

def load_data():
    """加载数据"""
    data = {}
    for filepath, (subject_id, gender) in FILE_MAPPING.items():
        df = pd.read_csv(filepath)
        # 清理列名（去除空格）
        df.columns = df.columns.str.strip()
        df = df[df['confidence'] > 0.8].reset_index(drop=True)
        data[subject_id] = {'df': df, 'gender': gender, 'subject_id': subject_id}
        print(f"✓ 加载 {subject_id} ({gender}): {len(df)} 帧")
    return data

def calculate_correlation_matrix(df, au_columns):
    """
    计算AU相关性矩阵，处理常数列（零方差）
    将NaN填充为0
    """
    # 提取AU数据
    au_data = df[au_columns].copy()
    
    # 计算相关性
    corr_matrix = au_data.corr()
    
    # 检查并记录常数列
    constant_cols = []
    for col in au_columns:
        if df[col].std() == 0 or df[col].nunique() == 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"    警告: 以下AU无变化（设为0相关）: {[c.replace('_r', '') for c in constant_cols]}")
    
    # 填充NaN为0（常数列的相关性）
    corr_matrix = corr_matrix.fillna(0)
    
    return corr_matrix, constant_cols

def plot_correlation_heatmap(corr_matrix, title, save_path, figsize=(14, 12)):
    """生成相关性热力图"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # 使用RdBu_r色图（红蓝反向）
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # 只显示下三角
    
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={'size': 8},
                ax=ax)
    
    ax.set_xticklabels(AU_SHORT_NAMES, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(AU_SHORT_NAMES, rotation=0, fontsize=10)
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 已保存: {os.path.basename(save_path)}")

def plot_full_correlation_matrix(corr_matrix, title, save_path, figsize=(16, 14)):
    """生成完整的相关性矩阵（显示所有数值）"""
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(corr_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={'size': 7},
                ax=ax)
    
    ax.set_xticklabels(AU_SHORT_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(AU_SHORT_NAMES, rotation=0, fontsize=9)
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    ✓ 已保存: {os.path.basename(save_path)}")

def analyze_strong_correlations(corr_matrix, subject_id, threshold=0.5):
    """分析强相关性AU对"""
    strong_corrs = []
    
    for i in range(len(AU_COLUMNS)):
        for j in range(i+1, len(AU_COLUMNS)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                strong_corrs.append({
                    'AU1': AU_SHORT_NAMES[i],
                    'AU2': AU_SHORT_NAMES[j],
                    'correlation': corr_val
                })
    
    # 按相关性强度排序
    strong_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    return strong_corrs

def main():
    print("=" * 80)
    print("积极情绪AU相关性矩阵热力图生成")
    print("=" * 80)
    
    # 创建输出目录
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    base_dir = f"/root/.openclaw/workspace/analysis_results/{today}_积极情绪_性别对比"
    corr_dir = os.path.join(base_dir, 'correlation_matrix')
    os.makedirs(corr_dir, exist_ok=True)
    
    print(f"\n📁 输出目录: {corr_dir}")
    
    # 加载数据
    print("\n📂 加载数据...")
    data = load_data()
    
    # 存储所有相关性矩阵用于性别对比
    all_corr_matrices = {}
    all_constant_cols = {}
    
    # 生成个人相关性热力图
    print("\n📊 生成个人AU相关性热力图...")
    for subject_id, info in data.items():
        df = info['df']
        gender = info['gender']
        
        corr_matrix, constant_cols = calculate_correlation_matrix(df, AU_COLUMNS)
        all_corr_matrices[subject_id] = corr_matrix
        all_constant_cols[subject_id] = constant_cols
        
        # 下三角版本
        plot_correlation_heatmap(
            corr_matrix,
            f'积极情绪 - {subject_id} ({gender}) - AU相关性矩阵 (下三角)',
            os.path.join(corr_dir, f'{subject_id}_correlation_lower.png')
        )
        
        # 完整矩阵版本
        plot_full_correlation_matrix(
            corr_matrix,
            f'积极情绪 - {subject_id} ({gender}) - AU相关性矩阵 (完整)',
            os.path.join(corr_dir, f'{subject_id}_correlation_full.png')
        )
    
    # 生成性别平均相关性
    print("\n📊 生成性别平均相关性热力图...")
    
    # 男性平均
    male_corr = (all_corr_matrices['M1'] + all_corr_matrices['M2']) / 2
    plot_correlation_heatmap(
        male_corr,
        '积极情绪 - 男性平均 (M1+M2)/2 - AU相关性矩阵',
        os.path.join(corr_dir, 'Male_Avg_correlation_lower.png')
    )
    
    # 女性
    female_corr = all_corr_matrices['F1']
    plot_correlation_heatmap(
        female_corr,
        '积极情绪 - 女性 (F1) - AU相关性矩阵',
        os.path.join(corr_dir, 'Female_F1_correlation_lower.png')
    )
    
    # 生成强相关性分析报告
    print("\n📊 生成强相关性分析报告...")
    report = []
    report.append("=" * 80)
    report.append("积极情绪AU强相关性分析报告 (|r| >= 0.5)")
    report.append("=" * 80)
    report.append("")
    
    for subject_id in ['M1', 'M2', 'F1']:
        gender = data[subject_id]['gender']
        report.append(f"【{subject_id} ({gender})】")
        
        if subject_id in all_constant_cols and all_constant_cols[subject_id]:
            report.append(f"  注意: 以下AU无变化: {[c.replace('_r', '') for c in all_constant_cols[subject_id]]}")
        
        strong_corrs = analyze_strong_correlations(all_corr_matrices[subject_id], subject_id)
        if strong_corrs:
            report.append(f"  强相关AU对 (共{len(strong_corrs)}对):")
            for i, corr_info in enumerate(strong_corrs[:10]):  # 只显示前10
                direction = "正相关" if corr_info['correlation'] > 0 else "负相关"
                report.append(f"    {corr_info['AU1']} - {corr_info['AU2']}: r={corr_info['correlation']:.3f} ({direction})")
            if len(strong_corrs) > 10:
                report.append(f"    ... 还有 {len(strong_corrs)-10} 对")
        else:
            report.append("  无强相关AU对 (|r| < 0.5)")
        report.append("")
    
    report.append("=" * 80)
    report_text = "\n".join(report)
    
    with open(os.path.join(corr_dir, 'correlation_analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    
    print(f"\n" + "=" * 80)
    print(f"✅ 相关性分析完成！结果保存在: {corr_dir}")
    print(f"=" * 80)
    
    # 列出生成的文件
    print("\n📋 生成的文件列表:")
    files = os.listdir(corr_dir)
    for f in sorted(files):
        print(f"  - {f}")

if __name__ == "__main__":
    main()
