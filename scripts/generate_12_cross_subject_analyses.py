#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为跨被试情绪对比生成完整图表
参照: 2025-02-17_悲伤情绪_2M1F对比
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

def generate_complete_analysis(analysis_name, subject_files, output_dir, is_cross_group=False):
    """
    生成完整的跨被试对比分析
    subject_files: {subject_id: (csv_file, gender/group_label)}
    """
    
    print(f"\n{'='*60}")
    print(f"分析: {analysis_name}")
    print(f"被试数: {len(subject_files)}")
    print(f"{'='*60}")
    
    # 创建输出目录
    subdirs = ['barplots', 'boxplots', 'radar', 'time_series', 'statistics', 'heatmaps']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # 读取所有被试数据
    subject_data = {}
    subject_info = {}
    all_au_cols = None
    
    for subject_id, (csv_file, label) in subject_files.items():
        if not os.path.exists(csv_file):
            print(f"  跳过: {csv_file} 不存在")
            continue
            
        df = pd.read_csv(csv_file, on_bad_lines='skip')
        subject_data[subject_id] = df
        subject_info[subject_id] = label
        
        if all_au_cols is None:
            all_au_cols = [c for c in df.columns if c.endswith('_r')]
        
        print(f"  {subject_id} ({label}): {len(df)} 帧")
    
    if len(subject_data) == 0:
        print("  无有效数据")
        return
    
    au_labels = [c.replace('_r', '').replace(' ', '') for c in all_au_cols]
    subjects = list(subject_data.keys())
    
    # 1. 计算统计数据
    stats_results = []
    for subject_id, df in subject_data.items():
        row = {'subject': subject_id, 'gender': subject_info[subject_id], 'n_frames': len(df)}
        for au in all_au_cols:
            row[au.strip()] = df[au].mean()
        stats_results.append(row)
    
    stats_df = pd.DataFrame(stats_results)
    stats_df.to_csv(os.path.join(output_dir, 'statistics', 'au_statistics.csv'), index=False)
    
    # 2. 生成AU激活热力图 (个体)
    print("\n  生成个体AU激活热力图...")
    for subject_id, df in subject_data.items():
        fig, ax = plt.subplots(figsize=(12, 2))
        data = df[all_au_cols].mean().values.reshape(1, -1)
        sns.heatmap(data, xticklabels=au_labels, yticklabels=[subject_id],
                   annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                   cbar_kws={'label': 'Intensity'})
        ax.set_title(f'{subject_id} - AU Activation', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmaps', f'heatmap_{subject_id}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # 3. 生成多被试对比热力图
    print("  生成多被试对比热力图...")
    fig, ax = plt.subplots(figsize=(14, len(subjects)*0.8+1))
    heatmap_data = stats_df[[c.strip() for c in all_au_cols]].values
    sns.heatmap(heatmap_data, xticklabels=au_labels, yticklabels=subjects,
               annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
               cbar_kws={'label': 'Mean Intensity'})
    ax.set_title('Subjects AU Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmaps', f'heatmap_{len(subjects)}subjects.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 性别/分组差异热力图 (如果适用)
    if not is_cross_group:
        # 性别对比
        genders = set(subject_info.values())
        if len(genders) > 1:
            print("  生成分组差异热力图...")
            gender_means = {}
            for gender in genders:
                gender_subjects = [s for s, g in subject_info.items() if g == gender]
                if gender_subjects:
                    gender_data = [subject_data[s][all_au_cols].mean() for s in gender_subjects]
                    gender_means[gender] = pd.concat(gender_data, axis=1).mean(axis=1)
            
            if len(gender_means) == 2:
                genders_list = list(gender_means.keys())
                diff = gender_means[genders_list[0]] - gender_means[genders_list[1]]
                
                fig, ax = plt.subplots(figsize=(14, 2))
                sns.heatmap(diff.values.reshape(1, -1), xticklabels=au_labels,
                           yticklabels=[f'{genders_list[0]} - {genders_list[1]}'],
                           annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
                ax.set_title('Group Difference', fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'heatmaps', 'heatmap_gender_diff.png'), 
                           dpi=150, bbox_inches='tight')
                plt.close()
    
    # 5. 柱状图 - 被试对比
    print("  生成柱状图...")
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(au_labels))
    width = 0.8 / len(subjects)
    
    for i, subject_id in enumerate(subjects):
        values = stats_df[stats_df['subject']==subject_id][[c.strip() for c in all_au_cols]].values[0]
        offset = (i - len(subjects)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=subject_id, alpha=0.8)
    
    ax.set_xlabel('Action Units', fontsize=12)
    ax.set_ylabel('Mean Intensity', fontsize=12)
    ax.set_title(f'{analysis_name} - Subjects Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(au_labels, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'barplots', f'barplot_{len(subjects)}subjects_comparison.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. 箱线图 - 关键AU分布
    print("  生成箱线图...")
    # 选择变异最大的5个AU
    au_vars = []
    for au_clean in au_labels:
        values = []
        for s in subjects:
            # 查找正确的列名
            col_name = None
            for col in subject_data[s].columns:
                if col.strip().replace('_r', '') == au_clean:
                    col_name = col
                    break
            if col_name:
                values.append(subject_data[s][col_name].values)
        if values and len(values) > 0:
            all_values = np.concatenate(values)
            if len(all_values) > 0:
                au_vars.append((au_clean, np.var(all_values)))
    
    if len(au_vars) == 0:
        print("  警告: 无法计算AU方差")
    else:
        top5_aus = sorted(au_vars, key=lambda x: x[1], reverse=True)[:5]
        top5_au_names = [au[0] for au in top5_aus]
        
        fig, axes = plt.subplots(1, min(5, len(top5_au_names)), figsize=(4*min(5, len(top5_au_names)), 4))
        if len(top5_au_names) == 1:
            axes = [axes]
        
        for idx, au_name in enumerate(top5_au_names):
            ax = axes[idx]
            data_to_plot = []
            for subject_id in subjects:
                # 查找正确的列名
                col_name = None
                for col in subject_data[subject_id].columns:
                    if col.strip().replace('_r', '') == au_name:
                        col_name = col
                        break
                if col_name:
                    data_to_plot.append(subject_data[subject_id][col_name].values)
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=[s[:6] for s in subjects], patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_alpha(0.7)
                ax.set_title(au_name, fontsize=11, fontweight='bold')
                ax.set_ylabel('Intensity')
                ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'{analysis_name} - Key AU Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'boxplots', 'boxplot_key_au_distribution.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # 7. 雷达图
    print("  生成雷达图...")
    # 选择Top 6 AU
    top6_aus = sorted(au_vars, key=lambda x: x[1], reverse=True)[:6]
    top6_au_names = [au[0] for au in top6_aus]
    
    angles = np.linspace(0, 2*np.pi, len(top6_au_names), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
    for i, subject_id in enumerate(subjects):
        values = []
        for au_name in top6_au_names:
            # 查找正确的列名
            col_name = None
            for col in subject_data[subject_id].columns:
                if col.strip().replace('_r', '') == au_name:
                    col_name = col
                    break
            if col_name:
                values.append(subject_data[subject_id][col_name].mean())
            else:
                values.append(0)
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=subject_id, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top6_au_names)
    ax.set_title(f'{analysis_name} - Individual Profiles', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar', 'radar_individual_profiles.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. 时间序列图 - 关键3个AU
    print("  生成时间序列图...")
    top3_aus = sorted(au_vars, key=lambda x: x[1], reverse=True)[:3]
    
    for au_name, _ in top3_aus:
        fig, axes = plt.subplots(len(subjects), 1, figsize=(14, 3*len(subjects)), sharex=True)
        if len(subjects) == 1:
            axes = [axes]
        
        colors_ts = plt.cm.tab10(np.linspace(0, 1, len(subjects)))
        
        for idx, subject_id in enumerate(subjects):
            ax = axes[idx]
            # 查找正确的列名
            col_name = None
            for col in subject_data[subject_id].columns:
                if col.strip().replace('_r', '') == au_name:
                    col_name = col
                    break
            
            if col_name:
                data = subject_data[subject_id][col_name].values
            else:
                data = np.array([0])
            
            # 降采样显示（如果数据点太多）
            if len(data) > 1000:
                step = len(data) // 1000
                x_vals = np.arange(0, len(data), step)
                y_vals = data[::step]
            else:
                x_vals = np.arange(len(data))
                y_vals = data
            
            ax.plot(x_vals, y_vals, color=colors_ts[idx], alpha=0.7, linewidth=0.8)
            ax.set_ylabel(f'{subject_id}\n({subject_info[subject_id]})', fontsize=10)
            ax.set_ylim(0, max(data) * 1.1 if max(data) > 0 else 0.1)
            ax.grid(True, alpha=0.3)
        
        axes[0].set_title(f'{au_name} Time Series', fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Frame', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'time_series', f'timeseries_{au_name}.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    # 9. 生成报告
    print("  生成分析报告...")
    generate_report(analysis_name, subjects, subject_info, stats_df, top5_aus, output_dir)
    
    print(f"  ✓ 完成: {output_dir}")

def generate_report(analysis_name, subjects, subject_info, stats_df, top_aus, output_dir):
    """生成分析报告"""
    
    report = f"""================================================================================
{analysis_name} 跨被试对比分析报告
日期: {datetime.now().strftime('%Y-%m-%d')}
================================================================================

1. 数据概况
----------------------------------------
情绪类型: {analysis_name.split('_')[-1].replace('情绪对比', '')}
数据来源: OpenFace 2.0 AU强度值

样本量:
"""
    
    for subject_id in subjects:
        n_frames = stats_df[stats_df['subject']==subject_id]['n_frames'].values[0]
        report += f"  • {subject_id} ({subject_info[subject_id]}): {n_frames} 帧\n"
    
    report += """
2. 核心发现
----------------------------------------
最具个体差异的AU (按方差排序):
"""
    
    for i, (au_name, var_val) in enumerate(top_aus[:5], 1):
        report += f"  {i}. {au_name}: var={var_val:.3f}\n"
    
    report += """
3. 被试特异性模式
----------------------------------------
"""
    
    for subject_id in subjects:
        subject_row = stats_df[stats_df['subject']==subject_id].iloc[0]
        au_cols = [c for c in stats_df.columns if c.startswith('AU')]
        top3 = sorted([(au, subject_row[au]) for au in au_cols], key=lambda x: x[1], reverse=True)[:3]
        
        report += f"\n{subject_id} ({subject_info[subject_id]}) 特征:\n"
        for au, val in top3:
            report += f"  • {au}: {val:.3f}\n"
    
    report += """
4. 输出文件清单
----------------------------------------
heatmaps/
  - heatmap_*.png                (个体AU激活热力图)
  - heatmap_*subjects.png        (多被试对比热力图)
  - heatmap_gender_diff.png      (分组差异热力图)
  - correlation_*.png            (17×17 AU相关性矩阵)

barplots/
  - barplot_*subjects_comparison.png

boxplots/
  - boxplot_key_au_distribution.png

radar/
  - radar_individual_profiles.png

time_series/
  - timeseries_AU*.png

statistics/
  - au_statistics.csv

================================================================================
"""
    
    with open(os.path.join(output_dir, 'analysis_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report)

# 定义12个分析任务
analyses = [
    # 正常组
    ("正常组_中性情绪对比", {
        'M1': ('../zcsj/M1_中性.csv', 'male'),
        'M2': ('../zcsj/M2_中性.csv', 'male'),
        'F1': ('../zcsj/F1_中性.csv', 'female'),
        'F2': ('../zcsj/F2_中性.csv', 'female')
    }),
    ("正常组_悲伤情绪对比", {
        'M1': ('../zcsj/M1_悲伤.csv', 'male'),
        'M2': ('../zcsj/M2_悲伤.csv', 'male'),
        'F1': ('../zcsj/F1_悲伤.csv', 'female'),
        'F2': ('../zcsj/F2_悲伤.csv', 'female')
    }),
    ("正常组_积极情绪对比", {
        'M1': ('../zcsj/M1_积极.csv', 'male'),
        'M2': ('../zcsj/M2_积极.csv', 'male'),
        'F1': ('../zcsj/F1_积极.csv', 'female'),
        'F2': ('../zcsj/F2_积极.csv', 'female')
    }),
    
    # 患者组
    ("患者组_中性情绪对比", {
        'ZFL': ('2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_中性.csv', 'patient'),
        'MHD': ('../sj/MHD_中性.csv', 'patient'),
        'WGL': ('../sj/WGL_中性.csv', 'patient'),
        'ZJK': ('../sj/ZJK_中性.csv', 'patient')
    }),
    ("患者组_悲伤情绪对比", {
        'ZFL': ('2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_悲伤.csv', 'patient'),
        'MHD': ('../sj/MHD_悲伤.csv', 'patient'),
        'WGL': ('../sj/WGL_悲伤.csv', 'patient'),
        'ZJK': ('../sj/ZJK_悲伤.csv', 'patient')
    }),
    ("患者组_积极情绪对比", {
        'ZFL': ('2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_积极.csv', 'patient'),
        'MHD': ('../sj/MHD_积极.csv', 'patient'),
        'WGL': ('../sj/WGL_积极.csv', 'patient'),
        'ZJK': ('../sj/ZJK_积极.csv', 'patient')
    }),
    
    # 跨组女性
    ("跨组女性_中性情绪对比", {
        'F1': ('../zcsj/F1_中性.csv', 'normal'),
        'F2': ('../zcsj/F2_中性.csv', 'normal'),
        'MHD': ('../sj/MHD_中性.csv', 'patient')
    }),
    ("跨组女性_悲伤情绪对比", {
        'F1': ('../zcsj/F1_悲伤.csv', 'normal'),
        'F2': ('../zcsj/F2_悲伤.csv', 'normal'),
        'MHD': ('../sj/MHD_悲伤.csv', 'patient')
    }),
    ("跨组女性_积极情绪对比", {
        'F1': ('../zcsj/F1_积极.csv', 'normal'),
        'F2': ('../zcsj/F2_积极.csv', 'normal'),
        'MHD': ('../sj/MHD_积极.csv', 'patient')
    }),
    
    # 跨组男性
    ("跨组男性_中性情绪对比", {
        'M1': ('../zcsj/M1_中性.csv', 'normal'),
        'M2': ('../zcsj/M2_中性.csv', 'normal'),
        'ZFL': ('2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_中性.csv', 'patient'),
        'WGL': ('../sj/WGL_中性.csv', 'patient'),
        'ZJK': ('../sj/ZJK_中性.csv', 'patient')
    }),
    ("跨组男性_悲伤情绪对比", {
        'M1': ('../zcsj/M1_悲伤.csv', 'normal'),
        'M2': ('../zcsj/M2_悲伤.csv', 'normal'),
        'ZFL': ('2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_悲伤.csv', 'patient'),
        'WGL': ('../sj/WGL_悲伤.csv', 'patient'),
        'ZJK': ('../sj/ZJK_悲伤.csv', 'patient')
    }),
    ("跨组男性_积极情绪对比", {
        'M1': ('../zcsj/M1_积极.csv', 'normal'),
        'M2': ('../zcsj/M2_积极.csv', 'normal'),
        'ZFL': ('2025-02-23_ZFL_三种情绪对比/raw_data/ZFL_积极.csv', 'patient'),
        'WGL': ('../sj/WGL_积极.csv', 'patient'),
        'ZJK': ('../sj/ZJK_积极.csv', 'patient')
    }),
]

if __name__ == '__main__':
    print("="*60)
    print("生成12个跨被试情绪对比完整分析")
    print("="*60)
    
    for i, (name, files) in enumerate(analyses, 1):
        output_dir = f"2025-02-23_{name}"
        is_cross = '跨组' in name
        print(f"\n[{i}/12] ", end="")
        generate_complete_analysis(name, files, output_dir, is_cross)
    
    print("\n" + "="*60)
    print("✓ 全部12个分析完成!")
    print("="*60)
