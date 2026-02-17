#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
积极情绪AU数据重组版分析脚本 (2男1女)
生成符合标准的重组目录结构
"""

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============ 配置 ============
# 文件映射: 文件路径 -> (被试ID, 性别)
FILE_MAPPING = {
    '/root/.openclaw/media/inbound/file_21---c1ecbaad-5700-42b7-a743-1b75f81b7ff1.csv': ('M1', 'Male'),
    '/root/.openclaw/media/inbound/file_22---772490a5-e791-43b9-8f4a-25c2f614570a.csv': ('M2', 'Male'),
    '/root/.openclaw/media/inbound/file_23---06535c58-c474-473b-a68d-aadcee3e3ca7.csv': ('F1', 'Female'),
}

# 17个AU (强度值)
AU_COLUMNS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

# AU中文名称
AU_NAMES_CN = {
    'AU01_r': 'AU01\n(眉毛内侧上扬)',
    'AU02_r': 'AU02\n(眉毛外侧上扬)',
    'AU04_r': 'AU04\n(眉毛下垂)',
    'AU05_r': 'AU05\n(上眼睑上扬)',
    'AU06_r': 'AU06\n(脸颊上扬)',
    'AU07_r': 'AU07\n(眼睑紧绷)',
    'AU09_r': 'AU09\n(鼻子皱起)',
    'AU10_r': 'AU10\n(上唇上扬)',
    'AU12_r': 'AU12\n(嘴角上扬)',
    'AU14_r': 'AU14\n(酒窝)',
    'AU15_r': 'AU15\n(嘴角下垂)',
    'AU17_r': 'AU17\n(下巴上扬)',
    'AU20_r': 'AU20\n(嘴唇横向伸展)',
    'AU23_r': 'AU23\n(嘴唇收紧)',
    'AU25_r': 'AU25\n(嘴唇分开)',
    'AU26_r': 'AU26\n(下颌下垂)',
    'AU45_r': 'AU45\n(眨眼)',
}

# ============ 数据加载 ============
def load_data():
    """加载并预处理数据"""
    data = {}
    for filepath, (subject_id, gender) in FILE_MAPPING.items():
        df = pd.read_csv(filepath)
        # 清理列名（去除空格）
        df.columns = df.columns.str.strip()
        # 只保留置信度>0.8的数据
        df = df[df['confidence'] > 0.8].reset_index(drop=True)
        data[subject_id] = {
            'df': df,
            'gender': gender,
            'subject_id': subject_id
        }
        print(f"✓ 加载 {subject_id} ({gender}): {len(df)} 帧")
    return data

# ============ 基础统计 ============
def calculate_basic_stats(data):
    """计算每个被试的基础统计"""
    stats_dict = {}
    for subject_id, info in data.items():
        df = info['df']
        subject_stats = {}
        for au in AU_COLUMNS:
            subject_stats[au] = {
                'mean': df[au].mean(),
                'std': df[au].std(),
                'max': df[au].max(),
                'min': df[au].min(),
            }
        stats_dict[subject_id] = subject_stats
    return stats_dict

# ============ 可视化函数 ============
def create_output_dirs(base_dir):
    """创建标准输出目录结构"""
    dirs = ['heatmaps', 'barplots', 'boxplots', 'radar', 'time_series', 'statistics', 'raw_data']
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    return {d: os.path.join(base_dir, d) for d in dirs}

def plot_individual_heatmaps(data, dirs):
    """生成个人AU热力图"""
    print("\n📊 生成个人AU激活热力图...")
    for subject_id, info in data.items():
        print(f"  处理 {subject_id}...")
        df = info['df']
        gender = info['gender']
        
        # 计算每100帧（约3秒）的平均值
        window_size = 100
        n_windows = len(df) // window_size
        
        heatmap_data = []
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window_mean = df.iloc[start:end][AU_COLUMNS].mean().values
            heatmap_data.append(window_mean)
        
        heatmap_data = np.array(heatmap_data).T
        print(f"    热力图数据形状: {heatmap_data.shape}")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=2.5)
        
        ax.set_yticks(range(len(AU_COLUMNS)))
        ax.set_yticklabels([AU_NAMES_CN[au] for au in AU_COLUMNS], fontsize=9)
        ax.set_xlabel('时间段 (约3秒/格)', fontsize=12)
        ax.set_title(f'积极情绪 - {subject_id} ({gender}) - AU激活强度热力图', fontsize=14, pad=20)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('AU强度', fontsize=11)
        
        plt.tight_layout()
        save_path = f"{dirs['heatmaps']}/{subject_id}_heatmap.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"    ✓ 热力图已保存: {save_path}")

def plot_gender_comparison_barplot(data, dirs):
    """生成性别对比柱状图"""
    print("\n📊 生成性别对比柱状图...")
    
    # 计算每个被试的平均AU激活
    subject_means = {}
    for subject_id, info in data.items():
        df = info['df']
        subject_means[subject_id] = [df[au].mean() for au in AU_COLUMNS]
    
    # 计算男性平均值
    male_mean = [(subject_means['M1'][i] + subject_means['M2'][i]) / 2 for i in range(len(AU_COLUMNS))]
    female_mean = subject_means['F1']
    
    # 计算男性内部差异
    male_diff = [abs(subject_means['M1'][i] - subject_means['M2'][i]) for i in range(len(AU_COLUMNS))]
    
    x = np.arange(len(AU_COLUMNS))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(16, 7))
    bars1 = ax.bar(x - width/2, male_mean, width, label='男性平均 (M1+M2)/2', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, female_mean, width, label='女性 (F1)', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('平均激活强度', fontsize=12)
    ax.set_title('积极情绪 - 性别AU激活对比', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([au.replace('_r', '') for au in AU_COLUMNS], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{dirs['barplots']}/gender_comparison_barplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 性别对比柱状图已保存")
    
    return male_mean, female_mean, male_diff

def plot_boxplots(data, dirs):
    """生成箱线图"""
    print("\n📊 生成箱线图...")
    
    fig, axes = plt.subplots(3, 6, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, au in enumerate(AU_COLUMNS):
        ax = axes[idx]
        
        box_data = []
        labels = []
        colors = []
        
        for subject_id, info in data.items():
            df = info['df']
            gender = info['gender']
            box_data.append(df[au].values)
            labels.append(f"{subject_id}\n({gender[:1]})")
            colors.append('#3498db' if gender == 'Male' else '#e74c3c')
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(AU_NAMES_CN[au].replace('\n', ' '), fontsize=9)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
    
    # 隐藏多余的子图
    for idx in range(len(AU_COLUMNS), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('积极情绪 - 各被试AU分布箱线图', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{dirs['boxplots']}/all_subjects_boxplots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 箱线图已保存")

def plot_radar_chart(data, dirs):
    """生成雷达图"""
    print("\n📊 生成雷达图...")
    
    # 计算平均值
    subject_means = {}
    for subject_id, info in data.items():
        df = info['df']
        subject_means[subject_id] = [df[au].mean() for au in AU_COLUMNS]
    
    male_mean = [(subject_means['M1'][i] + subject_means['M2'][i]) / 2 for i in range(len(AU_COLUMNS))]
    female_mean = subject_means['F1']
    
    # 选择前12个AU用于雷达图（避免过于拥挤）
    selected_aus = AU_COLUMNS[:12]
    angles = np.linspace(0, 2 * np.pi, len(selected_aus), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    male_values = male_mean[:12] + male_mean[:1]
    female_values = female_mean[:12] + female_mean[:1]
    
    ax.plot(angles, male_values, 'o-', linewidth=2, label='男性平均', color='#3498db')
    ax.fill(angles, male_values, alpha=0.25, color='#3498db')
    
    ax.plot(angles, female_values, 'o-', linewidth=2, label='女性', color='#e74c3c')
    ax.fill(angles, female_values, alpha=0.25, color='#e74c3c')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([au.replace('_r', '') for au in selected_aus], fontsize=10)
    ax.set_ylim(0, max(max(male_values), max(female_values)) * 1.2)
    ax.set_title('积极情绪 - 性别AU模式雷达图', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(f"{dirs['radar']}/gender_radar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 雷达图已保存")

def plot_time_series(data, dirs):
    """生成时间序列图"""
    print("\n📊 生成时间序列图...")
    
    # 选择关键AU
    key_aus = ['AU06_r', 'AU07_r', 'AU12_r', 'AU04_r']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, au in enumerate(key_aus):
        ax = axes[idx]
        
        for subject_id, info in data.items():
            df = info['df']
            gender = info['gender']
            color = '#3498db' if gender == 'Male' else '#e74c3c'
            linestyle = '-' if subject_id == 'M1' else ('--' if subject_id == 'M2' else '-.')
            
            # 降采样显示（每10帧取一点）
            timestamps = df['timestamp'][::10] if 'timestamp' in df.columns else np.arange(0, len(df), 10) / 30
            values = df[au][::10]
            
            ax.plot(timestamps, values, label=f'{subject_id} ({gender})', 
                   color=color, linestyle=linestyle, alpha=0.7, linewidth=1.2)
        
        ax.set_xlabel('时间 (秒)', fontsize=10)
        ax.set_ylabel('AU强度', fontsize=10)
        ax.set_title(f'{AU_NAMES_CN[au]} 时间序列', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.suptitle('积极情绪 - 关键AU时间序列', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{dirs['time_series']}/key_au_time_series.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ 时间序列图已保存")

# ============ 统计分析 ============
def statistical_analysis(data, dirs, male_mean, female_mean, male_diff):
    """生成统计分析报告"""
    print("\n📊 生成统计分析...")
    
    report = []
    report.append("=" * 80)
    report.append("积极情绪AU数据 - 统计分析报告")
    report.append("=" * 80)
    report.append("")
    
    # 1. 基本信息
    report.append("【1. 基本信息】")
    report.append(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"被试数量: 3人 (男性2人, 女性1人)")
    report.append(f"情绪类型: 积极情绪 (Positive/Happy)")
    report.append("")
    
    # 2. 个体内差异分析
    report.append("【2. 个体内AU激活均值】")
    for subject_id, info in data.items():
        df = info['df']
        report.append(f"\n{subject_id} ({info['gender']}):")
        au_means = [(au, df[au].mean()) for au in AU_COLUMNS]
        au_means.sort(key=lambda x: x[1], reverse=True)
        for au, mean_val in au_means[:5]:
            report.append(f"  {au}: {mean_val:.3f}")
    report.append("")
    
    # 3. 男性内部差异
    report.append("【3. 男性内部差异 (|M1-M2|)】")
    for i, au in enumerate(AU_COLUMNS):
        report.append(f"  {au}: {male_diff[i]:.3f}")
    report.append("")
    
    # 4. 性别差异
    report.append("【4. 性别差异 (男性平均 - 女性)】")
    gender_diffs = [(AU_COLUMNS[i], male_mean[i] - female_mean[i], male_mean[i], female_mean[i]) 
                    for i in range(len(AU_COLUMNS))]
    gender_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for au, diff, m_val, f_val in gender_diffs:
        direction = "男性>女性" if diff > 0 else "女性>男性"
        report.append(f"  {au}: {diff:+.3f} (男:{m_val:.3f}, 女:{f_val:.3f}) [{direction}]")
    report.append("")
    
    # 5. 关键发现
    report.append("【5. 关键发现】")
    top_diff = gender_diffs[0]
    report.append(f"• 最大性别差异AU: {top_diff[0]} (差异={top_diff[1]:.3f})")
    
    # 找出女性为0的AU
    zero_aus = [AU_COLUMNS[i] for i in range(len(AU_COLUMNS)) if female_mean[i] == 0]
    if zero_aus:
        report.append(f"• 女性无激活AU: {', '.join(zero_aus)}")
    
    # 积极情绪特有：检查AU12（嘴角上扬，微笑标志）
    au12_idx = AU_COLUMNS.index('AU12_r')
    report.append(f"• AU12 (微笑标志): 男性平均={male_mean[au12_idx]:.3f}, 女性={female_mean[au12_idx]:.3f}")
    
    report.append("")
    report.append("=" * 80)
    report.append("分析完成")
    report.append("=" * 80)
    
    # 保存报告
    report_text = "\n".join(report)
    with open(f"{dirs['statistics']}/analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text

def export_raw_data(data, dirs):
    """导出原始统计数据"""
    print("\n📊 导出原始数据...")
    
    # 导出每个被试的AU均值
    stats_df = pd.DataFrame()
    for subject_id, info in data.items():
        df = info['df']
        subject_stats = {'subject_id': subject_id, 'gender': info['gender']}
        for au in AU_COLUMNS:
            subject_stats[au.replace('_r', '_mean')] = df[au].mean()
            subject_stats[au.replace('_r', '_std')] = df[au].std()
        stats_df = pd.concat([stats_df, pd.DataFrame([subject_stats])], ignore_index=True)
    
    stats_df.to_csv(f"{dirs['raw_data']}/subject_statistics.csv", index=False, encoding='utf-8-sig')
    print("  ✓ 统计数据已导出")
    
    # 导出性别对比数据
    gender_df = pd.DataFrame({
        'AU': [au.replace('_r', '') for au in AU_COLUMNS],
        'Male_M1': [data['M1']['df'][au].mean() for au in AU_COLUMNS],
        'Male_M2': [data['M2']['df'][au].mean() for au in AU_COLUMNS],
        'Male_Avg': [((data['M1']['df'][au].mean() + data['M2']['df'][au].mean()) / 2) for au in AU_COLUMNS],
        'Female_F1': [data['F1']['df'][au].mean() for au in AU_COLUMNS],
        'Gender_Diff(M-F)': [((data['M1']['df'][au].mean() + data['M2']['df'][au].mean()) / 2 - data['F1']['df'][au].mean()) for au in AU_COLUMNS],
    })
    gender_df.to_csv(f"{dirs['raw_data']}/gender_comparison.csv", index=False, encoding='utf-8-sig')
    print("  ✓ 性别对比数据已导出")

# ============ 主函数 ============
def main():
    print("=" * 80)
    print("积极情绪AU数据分析 (重组版)")
    print("=" * 80)
    
    # 创建输出目录
    from datetime import datetime
    today = datetime.now().strftime('%Y-%m-%d')
    base_dir = f"/root/.openclaw/workspace/analysis_results/{today}_积极情绪_性别对比"
    dirs = create_output_dirs(base_dir)
    print(f"\n📁 输出目录: {base_dir}")
    
    # 加载数据
    print("\n📂 加载数据...")
    data = load_data()
    
    # 计算基础统计
    stats_dict = calculate_basic_stats(data)
    
    # 生成可视化
    plot_individual_heatmaps(data, dirs)
    male_mean, female_mean, male_diff = plot_gender_comparison_barplot(data, dirs)
    plot_boxplots(data, dirs)
    plot_radar_chart(data, dirs)
    plot_time_series(data, dirs)
    
    # 统计分析
    report = statistical_analysis(data, dirs, male_mean, female_mean, male_diff)
    
    # 导出数据
    export_raw_data(data, dirs)
    
    print(f"\n" + "=" * 80)
    print(f"✅ 分析完成！所有结果保存在: {base_dir}")
    print(f"=" * 80)
    
    # 列出生成的文件
    print("\n📋 生成的文件列表:")
    for dir_name, dir_path in dirs.items():
        files = os.listdir(dir_path) if os.path.exists(dir_path) else []
        if files:
            print(f"  📂 {dir_name}/")
            for f in files[:5]:
                print(f"     - {f}")
            if len(files) > 5:
                print(f"     ... 还有 {len(files)-5} 个文件")

if __name__ == "__main__":
    main()
