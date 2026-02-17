#!/usr/bin/env python3
"""
数据可视化脚本
- 热力图（相关性矩阵）
- 相关散点图
- 组间对比图

用法：python data_visualization.py <数据文件>
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """加载数据，支持CSV和Excel"""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    else:
        raise ValueError("不支持的文件格式，请使用 CSV 或 Excel")

def plot_correlation_heatmap(df, columns=None, save_path='heatmap.png'):
    """
    绘制相关性热力图
    
    参数:
        df: DataFrame
        columns: 要分析的列名列表，None则使用所有数值列
        save_path: 保存路径
    """
    # 选择数值列
    if columns:
        numeric_df = df[columns]
    else:
        numeric_df = df.select_dtypes(include=[np.number])
    
    # 计算相关性矩阵
    corr_matrix = numeric_df.corr()
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                annot=True,           # 显示数值
                cmap='RdBu_r',        # 红蓝配色
                center=0,             # 以0为中心
                square=True,          # 方形
                linewidths=0.5,       # 线宽
                fmt='.2f',            # 数值格式
                cbar_kws={"shrink": 0.8})
    plt.title('相关性热力图', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"热力图已保存: {save_path}")

def plot_scatter(df, x_col, y_col, hue_col=None, save_path='scatter.png'):
    """
    绘制相关散点图
    
    参数:
        df: DataFrame
        x_col: X轴列名
        y_col: Y轴列名
        hue_col: 分组列名（可选）
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 7))
    
    if hue_col:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, s=100, alpha=0.7)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, s=100, alpha=0.7)
    
    # 添加趋势线
    sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color='red', 
                line_kws={'linestyle': '--', 'alpha': 0.8})
    
    # 计算并显示相关系数
    corr = df[x_col].corr(df[y_col])
    plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.title(f'{x_col} vs {y_col} 散点图', fontsize=14, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"散点图已保存: {save_path}")

def plot_group_comparison(df, value_col, group_col, save_path='group_comparison.png'):
    """
    绘制组间对比图（箱线图 + 散点）
    
    参数:
        df: DataFrame
        value_col: 数值列名
        group_col: 分组列名
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 箱线图
    sns.boxplot(data=df, x=group_col, y=value_col, ax=axes[0], palette='Set2')
    axes[0].set_title('组间箱线图对比', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(group_col, fontsize=11)
    axes[0].set_ylabel(value_col, fontsize=11)
    
    # 添加散点
    sns.stripplot(data=df, x=group_col, y=value_col, ax=axes[0], 
                  color='black', alpha=0.5, size=6)
    
    # 柱状图（均值±标准差）
    summary = df.groupby(group_col)[value_col].agg(['mean', 'std']).reset_index()
    bars = axes[1].bar(summary[group_col], summary['mean'], 
                       yerr=summary['std'], capsize=5, 
                       color='skyblue', edgecolor='navy', alpha=0.7)
    axes[1].set_title('组间均值对比 (Mean ± SD)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(group_col, fontsize=11)
    axes[1].set_ylabel(f'{value_col} (Mean ± SD)', fontsize=11)
    
    # 在柱子上添加数值标签
    for bar, mean, std in zip(bars, summary['mean'], summary['std']):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std,
                     f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"组间对比图已保存: {save_path}")

def main():
    if len(sys.argv) < 2:
        print("用法: python data_visualization.py <数据文件>")
        print("示例: python data_visualization.py data.csv")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    # 加载数据
    print(f"正在加载数据: {filepath}")
    df = load_data(filepath)
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print("\n数据预览:")
    print(df.head())
    
    # 获取数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n数值列: {numeric_cols}")
    
    # 1. 热力图
    if len(numeric_cols) >= 2:
        plot_correlation_heatmap(df, numeric_cols, 'correlation_heatmap.png')
    
    # 2. 散点图（如果至少有2个数值列）
    if len(numeric_cols) >= 2:
        plot_scatter(df, numeric_cols[0], numeric_cols[1], save_path='scatter_plot.png')
    
    # 3. 组间对比（如果有分组列）
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols and numeric_cols:
        plot_group_comparison(df, numeric_cols[0], categorical_cols[0], 'group_comparison.png')
    
    print("\n✅ 所有图表生成完成!")

if __name__ == '__main__':
    main()
