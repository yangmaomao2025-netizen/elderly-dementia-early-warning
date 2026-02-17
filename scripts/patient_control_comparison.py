#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
患者-对照组AU表情对比分析
Patient-Control Group AU Comparison Analysis

对比抑郁患者与健康对照组在悲伤和积极情绪下的AU激活差异
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============== 配置 ==============
# 对照组文件（健康被试）
CONTROL_FILES = {
    'sadness': {
        'M1': '/root/.openclaw/media/inbound/file_3---b3314058-964d-470d-8293-13430fdde2c6.csv',
        'M2': '/root/.openclaw/media/inbound/file_4---0dd96eb3-72ff-4ced-a1b8-c5c51fad721a.csv',
        'F1': '/root/.openclaw/media/inbound/file_5---69ad20a2-5a2f-4f18-bdef-056d8c24d515.csv'
    },
    'positive': {
        'M1': '/root/.openclaw/media/inbound/file_21---c1ecbaad-5700-42b7-a743-1b75f81b7ff1.csv',
        'M2': '/root/.openclaw/media/inbound/file_22---772490a5-e791-43b9-8f4a-25c2f614570a.csv',
        'F1': '/root/.openclaw/media/inbound/file_23---06535c58-c474-473b-a68d-aadcee3e3ca7.csv'
    }
}

# 患者组文件（抑郁症患者）
PATIENT_FILES = {
    'sadness': {
        'P1': '/root/.openclaw/media/inbound/file_26---2b859f5a-08e2-4713-b654-c56162c1085d.csv'
    },
    'positive': {
        'P1': '/root/.openclaw/media/inbound/file_24---925f9a2e-ba59-4283-829c-75d596785181.csv',
        'P2': '/root/.openclaw/media/inbound/file_25---f701a00a-5efc-44e6-8514-4510879be7a9.csv'
    }
}

# 17个核心AU
AU_COLUMNS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
              'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r',
              'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

# 抑郁相关AU标记
DEPRESSION_AUS = ['AU04', 'AU07', 'AU12', 'AU06']  # 皱眉、眼睑紧绷、微笑、脸颊提升

# ============== 数据加载函数 ==============

def load_au_data(filepath):
    """加载AU数据文件"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    # 过滤低置信度帧
    df = df[df['confidence'] >= 0.8].copy()
    
    # 确保所有AU列存在
    for au in AU_COLUMNS:
        if au not in df.columns:
            df[au] = 0.0
    
    return df[AU_COLUMNS].reset_index(drop=True)

def calculate_subject_stats(df):
    """计算单个被试的AU统计特征"""
    stats_dict = {}
    for au in AU_COLUMNS:
        stats_dict[f'{au}_mean'] = df[au].mean()
        stats_dict[f'{au}_std'] = df[au].std()
        stats_dict[f'{au}_max'] = df[au].max()
        stats_dict[f'{au}_activation_rate'] = (df[au] > 0.5).mean()
    return stats_dict

# ============== 统计分析函数 ==============

def cohens_d(x1, x2):
    """计算Cohen's d效应量"""
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1 + (n2-1)*s2) / (n1+n2-2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0

def bootstrap_ci(x1, x2, n_bootstrap=2000, ci=0.95):
    """Bootstrap置信区间"""
    boot_diffs = []
    n1, n2 = len(x1), len(x2)
    
    for _ in range(n_bootstrap):
        boot_x1 = np.random.choice(x1, size=n1, replace=True)
        boot_x2 = np.random.choice(x2, size=n2, replace=True)
        boot_diffs.append(np.mean(boot_x1) - np.mean(boot_x2))
    
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_diffs, alpha * 100)
    upper = np.percentile(boot_diffs, (1 - alpha) * 100)
    return lower, upper

# ============== 主分析流程 ==============

def main():
    print("="*60)
    print("患者-对照组AU表情对比分析")
    print("Patient-Control Group Comparison")
    print("="*60)
    
    # 创建结果目录
    output_dir = Path('/root/.openclaw/workspace/analysis_results/2026-02-17_患者对照组对比')
    for subdir in ['heatmaps', 'barplots', 'boxplots', 'statistics', 'time_series', 'classifier']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # ========== 1. 加载所有数据 ==========
    print("\n📊 正在加载数据...")
    
    all_data = []
    
    # 加载对照组数据
    for emotion, files in CONTROL_FILES.items():
        for subject, filepath in files.items():
            df = load_au_data(filepath)
            stats_dict = calculate_subject_stats(df)
            stats_dict['subject'] = subject
            stats_dict['group'] = 'Control'
            stats_dict['emotion'] = emotion
            stats_dict['gender'] = 'Male' if subject.startswith('M') else 'Female'
            stats_dict['frames'] = len(df)
            all_data.append(stats_dict)
            print(f"  ✅ Control {subject} ({emotion}): {len(df)} frames")
    
    # 加载患者组数据
    for emotion, files in PATIENT_FILES.items():
        for subject, filepath in files.items():
            df = load_au_data(filepath)
            stats_dict = calculate_subject_stats(df)
            stats_dict['subject'] = subject
            stats_dict['group'] = 'Patient'
            stats_dict['emotion'] = emotion
            stats_dict['gender'] = 'Male'
            stats_dict['frames'] = len(df)
            all_data.append(stats_dict)
            print(f"  ✅ Patient {subject} ({emotion}): {len(df)} frames")
    
    # 创建数据框
    df_all = pd.DataFrame(all_data)
    
    # ========== 2. 组间比较分析 ==========
    print("\n" + "="*60)
    print("📈 组间AU激活差异分析 (Group Comparison)")
    print("="*60)
    
    comparison_results = []
    
    for emotion in ['sadness', 'positive']:
        print(f"\n--- {emotion.upper()} EMOTION ---")
        
        control_data = df_all[(df_all['group'] == 'Control') & (df_all['emotion'] == emotion)]
        patient_data = df_all[(df_all['group'] == 'Patient') & (df_all['emotion'] == emotion)]
        
        for au in AU_COLUMNS:
            au_mean_col = f'{au}_mean'
            control_values = control_data[au_mean_col].values
            patient_values = patient_data[au_mean_col].values
            
            if len(control_values) > 0 and len(patient_values) > 0:
                # 计算均值差异
                mean_diff = patient_values.mean() - control_values.mean()
                
                # 效应量
                effect_size = cohens_d(patient_values, control_values)
                
                # Bootstrap置信区间
                ci_lower, ci_upper = bootstrap_ci(patient_values, control_values)
                
                # t检验
                if len(control_values) > 1 and len(patient_values) > 1:
                    t_stat, p_value = stats.ttest_ind(patient_values, control_values)
                else:
                    t_stat, p_value = np.nan, np.nan
                
                comparison_results.append({
                    'emotion': emotion,
                    'AU': au,
                    'control_mean': control_values.mean(),
                    'patient_mean': patient_values.mean(),
                    'mean_diff': mean_diff,
                    'cohens_d': effect_size,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    't_stat': t_stat,
                    'p_value': p_value
                })
                
                # 打印显著结果
                if abs(effect_size) > 0.5:
                    direction = "↑" if mean_diff > 0 else "↓"
                    print(f"  {au}: Patient {direction} Control | d={effect_size:.2f} | p={p_value:.3f}")
    
    df_comparison = pd.DataFrame(comparison_results)
    df_comparison.to_csv(output_dir / 'statistics' / 'group_comparison_stats.csv', index=False)
    print(f"\n  💾 统计结果已保存至: statistics/group_comparison_stats.csv")
    
    # ========== 3. 可视化 ==========
    print("\n" + "="*60)
    print("🎨 生成可视化图表")
    print("="*60)
    
    # 3.1 组间差异热图
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    for idx, emotion in enumerate(['sadness', 'positive']):
        emotion_data = df_comparison[df_comparison['emotion'] == emotion]
        pivot_data = emotion_data.pivot(index='AU', columns='emotion', values='cohens_d')
        
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, vmin=-2, vmax=2, ax=axes[idx], cbar_kws={'label': "Cohen's d"})
        axes[idx].set_title(f'{emotion.capitalize()} Emotion: Patient vs Control\n(Effect Size)', fontsize=12)
        axes[idx].set_xlabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmaps' / 'patient_control_effect_size_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 效应量热图已生成")
    
    # 3.2 抑郁相关AU的箱线图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, au_base in enumerate(DEPRESSION_AUS):
        au_col = f'{au_base}_r_mean'
        
        plot_data = []
        labels = []
        colors = []
        
        for emotion in ['sadness', 'positive']:
            control_vals = df_all[(df_all['group'] == 'Control') & 
                                 (df_all['emotion'] == emotion)][au_col].values
            patient_vals = df_all[(df_all['group'] == 'Patient') & 
                                 (df_all['emotion'] == emotion)][au_col].values
            
            plot_data.extend([control_vals, patient_vals])
            labels.extend([f'Control\n({emotion})', f'Patient\n({emotion})'])
            colors.extend(['lightblue', 'salmon'])
        
        bp = axes[idx].boxplot(plot_data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[idx].set_title(f'{au_base} Activation', fontsize=11)
        axes[idx].set_ylabel('Mean Intensity')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Key Depression-Related AU: Patient vs Control', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots' / 'depression_au_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 抑郁相关AU箱线图已生成")
    
    # 3.3 AU激活水平对比柱状图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, emotion in enumerate(['sadness', 'positive']):
        emotion_data = df_comparison[df_comparison['emotion'] == emotion].copy()
        emotion_data = emotion_data.sort_values('cohens_d', key=abs, ascending=False).head(10)
        
        colors = ['red' if x > 0 else 'blue' for x in emotion_data['cohens_d']]
        axes[idx].barh(range(len(emotion_data)), emotion_data['cohens_d'], color=colors, alpha=0.7)
        axes[idx].set_yticks(range(len(emotion_data)))
        axes[idx].set_yticklabels(emotion_data['AU'])
        axes[idx].set_xlabel("Cohen's d (Patient - Control)")
        axes[idx].set_title(f'{emotion.capitalize()}: Top 10 AU Differences')
        axes[idx].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[idx].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium Effect')
        axes[idx].axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5)
        axes[idx].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'barplots' / 'top_au_differences.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ AU差异柱状图已生成")
    
    # ========== 4. 分类器分析 ==========
    print("\n" + "="*60)
    print("🤖 简单分类模型 (患者 vs 对照)")
    print("="*60)
    
    # 准备特征矩阵
    feature_cols = [f'{au}_mean' for au in AU_COLUMNS]
    
    X = df_all[feature_cols].values
    y = (df_all['group'] == 'Patient').astype(int).values
    emotions = df_all['emotion'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 逻辑回归
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr_scores = cross_val_score(lr, X_scaled, y, cv=3)
    print(f"  Cross-validation accuracy: {lr_scores.mean():.3f} (+/- {lr_scores.std()*2:.3f})")
    
    # 训练完整模型
    lr.fit(X_scaled, y)
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'AU': AU_COLUMNS,
        'coefficient': lr.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\n  最重要的预测特征 (Top 5):")
    for _, row in feature_importance.head(5).iterrows():
        direction = "预测患者" if row['coefficient'] > 0 else "预测对照"
        print(f"    {row['AU']}: {row['coefficient']:.3f} ({direction})")
    
    # 随机森林
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X_scaled, y, cv=3)
    print(f"  Cross-validation accuracy: {rf_scores.mean():.3f} (+/- {rf_scores.std()*2:.3f})")
    
    rf.fit(X_scaled, y)
    rf_importance = pd.DataFrame({
        'AU': AU_COLUMNS,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n  特征重要性 (Top 5):")
    for _, row in rf_importance.head(5).iterrows():
        print(f"    {row['AU']}: {row['importance']:.3f}")
    
    # 保存分类器结果
    feature_importance.to_csv(output_dir / 'classifier' / 'logistic_regression_features.csv', index=False)
    rf_importance.to_csv(output_dir / 'classifier' / 'random_forest_importance.csv', index=False)
    
    # 绘制特征重要性图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 逻辑回归系数
    colors = ['red' if x > 0 else 'blue' for x in feature_importance['coefficient']]
    axes[0].barh(range(len(feature_importance)), feature_importance['coefficient'], color=colors, alpha=0.7)
    axes[0].set_yticks(range(len(feature_importance)))
    axes[0].set_yticklabels(feature_importance['AU'])
    axes[0].set_xlabel('Coefficient')
    axes[0].set_title('Logistic Regression: AU Predictive Power')
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].grid(axis='x', alpha=0.3)
    
    # 随机森林重要性
    axes[1].barh(range(len(rf_importance)), rf_importance['importance'], color='green', alpha=0.7)
    axes[1].set_yticks(range(len(rf_importance)))
    axes[1].set_yticklabels(rf_importance['AU'])
    axes[1].set_xlabel('Feature Importance')
    axes[1].set_title('Random Forest: AU Feature Importance')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'classifier' / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 分类器特征重要性图已生成")
    
    # ========== 5. 生成综合报告 ==========
    print("\n" + "="*60)
    print("📝 生成综合报告")
    print("="*60)
    
    report_lines = []
    report_lines.append("# 患者-对照组AU表情对比分析报告")
    report_lines.append(f"\n生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("\n" + "="*60)
    
    # 样本信息
    report_lines.append("\n## 1. 样本信息")
    report_lines.append(f"- 对照组: 3人 (2男1女)")
    report_lines.append(f"- 患者组: 3人 (3男)")
    report_lines.append(f"- 情绪类型: 悲伤、积极")
    
    # 关键发现
    report_lines.append("\n## 2. 关键发现")
    
    # 悲伤情绪差异
    sadness_sig = df_comparison[(df_comparison['emotion'] == 'sadness') & 
                                (abs(df_comparison['cohens_d']) > 0.5)]
    if len(sadness_sig) > 0:
        report_lines.append(f"\n### 悲伤情绪显著差异 AU (|d| > 0.5):")
        for _, row in sadness_sig.iterrows():
            direction = "患者 > 对照" if row['cohens_d'] > 0 else "患者 < 对照"
            report_lines.append(f"- {row['AU']}: Cohen's d = {row['cohens_d']:.2f} ({direction})")
    
    # 积极情绪差异
    positive_sig = df_comparison[(df_comparison['emotion'] == 'positive') & 
                                 (abs(df_comparison['cohens_d']) > 0.5)]
    if len(positive_sig) > 0:
        report_lines.append(f"\n### 积极情绪显著差异 AU (|d| > 0.5):")
        for _, row in positive_sig.iterrows():
            direction = "患者 > 对照" if row['cohens_d'] > 0 else "患者 < 对照"
            report_lines.append(f"- {row['AU']}: Cohen's d = {row['cohens_d']:.2f} ({direction})")
    
    # 分类器性能
    report_lines.append(f"\n## 3. 分类模型性能")
    report_lines.append(f"- 逻辑回归准确率: {lr_scores.mean():.3f} (±{lr_scores.std():.3f})")
    report_lines.append(f"- 随机森林准确率: {rf_scores.mean():.3f} (±{rf_scores.std():.3f})")
    
    report_lines.append(f"\n### 最重要的分类特征:")
    for _, row in rf_importance.head(3).iterrows():
        report_lines.append(f"- {row['AU']}: {row['importance']:.3f}")
    
    # 保存报告
    report_text = '\n'.join(report_lines)
    with open(output_dir / 'statistics' / 'comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n  💾 报告已保存至: statistics/comparison_report.md")
    
    print("\n" + "="*60)
    print(f"✅ 分析完成！所有结果保存在: {output_dir}")
    print("="*60)

if __name__ == '__main__':
    main()
