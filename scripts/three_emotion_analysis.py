#!/usr/bin/env python3
"""
三情绪AU特征对比分析脚本
数据: OpenFace 2.0 输出 (AU强度值)
对比: 悲伤 vs 风景(中性) vs 正性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal
import warnings
warnings.filterwarnings('ignore')

# 设置图形样式
plt.style.use('default')
sns.set_palette("husl")

# ============ 配置路径和情绪标签 ============
# 请根据实际文件修改以下配置
FILE_CONFIG = {
    '悲伤': '/root/.openclaw/media/inbound/file_3---b3314058-964d-470d-8293-13430fdde2c6.csv',
    '风景': '/root/.openclaw/media/inbound/file_4---0dd96eb3-72ff-4ced-a1b8-c5c51fad721a.csv',
    '正性': '/root/.openclaw/media/inbound/file_5---69ad20a2-5a2f-4f18-bdef-056d8c24d515.csv'
}

# 情绪颜色方案
EMOTION_COLORS = {
    '悲伤': '#3498db',  # 蓝色
    '风景': '#95a5a6',  # 灰色
    '正性': '#e74c3c'   # 红色
}

# 核心AU列表 (排除 _c 列，只保留强度值 _r)
CORE_AU = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
           'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
           'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

def load_and_preprocess(filepath, emotion_label):
    """加载数据并预处理"""
    df = pd.read_csv(filepath)
    
    # 清理列名（去除空格）
    df.columns = df.columns.str.strip()
    
    # 只保留成功检测的帧
    df = df[df['success'] == 1]
    df = df[df['confidence'] >= 0.95]  # 高置信度
    
    # 添加情绪标签
    df['emotion'] = emotion_label
    
    return df

def cohen_d(x, y):
    """计算Cohen's d效应量"""
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        ((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof
    )

def perform_anova(data_dict, au_column):
    """执行单因素ANOVA"""
    groups = [df[au_column].values for df in data_dict.values()]
    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value

def perform_kruskal(data_dict, au_column):
    """执行Kruskal-Wallis检验（非参数）"""
    groups = [df[au_column].values for df in data_dict.values()]
    h_stat, p_value = kruskal(*groups)
    return h_stat, p_value

def bonferroni_posthoc(data_dict, au_column, alpha=0.05):
    """Bonferroni校正事后检验"""
    emotions = list(data_dict.keys())
    n_comparisons = len(emotions) * (len(emotions) - 1) // 2
    corrected_alpha = alpha / n_comparisons
    
    results = []
    for i in range(len(emotions)):
        for j in range(i+1, len(emotions)):
            group1 = data_dict[emotions[i]][au_column].values
            group2 = data_dict[emotions[j]][au_column].values
            
            # t检验
            t_stat, p_val = stats.ttest_ind(group1, group2)
            
            # 效应量
            effect = cohen_d(group1, group2)
            
            results.append({
                'comparison': f"{emotions[i]} vs {emotions[j]}",
                't_stat': t_stat,
                'p_value': p_val,
                'p_corrected': p_val * n_comparisons,
                'significant': p_val < corrected_alpha,
                'cohens_d': effect,
                'effect_size': 'Large' if abs(effect) >= 0.8 else ('Medium' if abs(effect) >= 0.5 else 'Small')
            })
    
    return pd.DataFrame(results)

# ============ 主程序 ============
print("=" * 70)
print("OpenFace AU情绪特征对比分析")
print("=" * 70)

# 1. 加载数据
print("\n📂 加载数据文件...")
data_dict = {}
for emotion, filepath in FILE_CONFIG.items():
    df = load_and_preprocess(filepath, emotion)
    data_dict[emotion] = df
    print(f"  {emotion}: {len(df)} 帧 (置信度≥0.95)")

# 2. 描述性统计
print("\n" + "=" * 70)
print("📊 描述性统计 (AU平均强度 ± 标准差)")
print("=" * 70)

desc_stats = []
for emotion, df in data_dict.items():
    stats_row = {'Emotion': emotion, 'N_Frames': len(df)}
    for au in CORE_AU:
        stats_row[f"{au}_mean"] = df[au].mean()
        stats_row[f"{au}_std"] = df[au].std()
    desc_stats.append(stats_row)

desc_df = pd.DataFrame(desc_stats)

# 打印主要AU的均值
print("\n主要AU强度对比:")
main_au_display = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU15_r', 'AU25_r']
for au in main_au_display:
    print(f"\n{au}:")
    for _, row in desc_df.iterrows():
        mean_val = row[f"{au}_mean"]
        std_val = row[f"{au}_std"]
        print(f"  {row['Emotion']:6s}: {mean_val:.3f} ± {std_val:.3f}")

# 3. 组间差异检验
print("\n" + "=" * 70)
print("📈 组间差异检验 (ANOVA)")
print("=" * 70)

anova_results = []
for au in CORE_AU:
    f_stat, p_val = perform_anova(data_dict, au)
    anova_results.append({
        'AU': au,
        'F_statistic': f_stat,
        'p_value': p_val,
        'significant': p_val < 0.05,
        'significant_bonferroni': p_val < (0.05 / len(CORE_AU))  # Bonferroni校正
    })

anova_df = pd.DataFrame(anova_results)
anova_df = anova_df.sort_values('p_value')

print("\n显著的AU差异 (p < 0.05):")
sig_au = anova_df[anova_df['significant']]
if len(sig_au) > 0:
    for _, row in sig_au.head(10).iterrows():
        sig_marker = "***" if row['significant_bonferroni'] else "**" if row['p_value'] < 0.01 else "*"
        print(f"  {row['AU']:8s}: F={row['F_statistic']:6.2f}, p={row['p_value']:.4f} {sig_marker}")
else:
    print("  无显著差异")

# 4. 事后检验 (对最显著的AU)
print("\n" + "=" * 70)
print("🔍 事后检验 (Bonferroni校正) - 最显著的AU")
print("=" * 70)

if len(sig_au) > 0:
    top_au = sig_au.iloc[0]['AU']
    print(f"\n{top_au} 的组间对比:")
    posthoc = bonferroni_posthoc(data_dict, top_au)
    for _, row in posthoc.iterrows():
        sig = "***" if row['significant'] else ""
        print(f"  {row['comparison']:15s}: t={row['t_stat']:6.2f}, "
              f"p={row['p_value']:.4f}, Cohen's d={row['cohens_d']:.3f} "
              f"({row['effect_size']}) {sig}")

# 5. 可视化
print("\n📊 生成可视化图表...")

fig = plt.figure(figsize=(20, 24))

# 5.1 各情绪AU平均强度柱状图
ax1 = plt.subplot(4, 2, 1)
x_pos = np.arange(len(CORE_AU))
width = 0.25

for i, (emotion, df) in enumerate(data_dict.items()):
    means = [df[au].mean() for au in CORE_AU]
    sems = [df[au].sem() for au in CORE_AU]  # 标准误
    ax1.bar(x_pos + i*width, means, width, yerr=sems, 
            label=emotion, color=EMOTION_COLORS[emotion], alpha=0.8, capsize=3)

ax1.set_xlabel('Action Units', fontsize=11)
ax1.set_ylabel('Mean Intensity', fontsize=11)
ax1.set_title('AU Mean Intensity by Emotion (± SEM)', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos + width)
ax1.set_xticklabels([au.replace('_r', '') for au in CORE_AU], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 5.2 箱线图对比
ax2 = plt.subplot(4, 2, 2)
plot_data = []
plot_labels = []
for emotion, df in data_dict.items():
    # 选择几个关键AU
    for au in ['AU04_r', 'AU06_r', 'AU12_r', 'AU25_r']:
        plot_data.extend(df[au].values)
        plot_labels.extend([f"{emotion}\n{au}"] * len(df))

box_df = pd.DataFrame({'Value': plot_data, 'Group': plot_labels})
sns.boxplot(data=box_df, x='Group', y='Value', ax=ax2, palette='Set2')
ax2.set_title('Key AU Distribution by Emotion', fontsize=13, fontweight='bold')
ax2.set_xlabel('')
ax2.tick_params(axis='x', rotation=45)

# 5.3 悲伤情绪AU相关性热图
ax3 = plt.subplot(4, 3, 7)
sad_corr = data_dict['悲伤'][CORE_AU].corr()
sns.heatmap(sad_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax3, cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
ax3.set_title('Sad - AU Correlation', fontsize=12, fontweight='bold')

# 5.4 风景情绪AU相关性热图
ax4 = plt.subplot(4, 3, 8)
neu_corr = data_dict['风景'][CORE_AU].corr()
sns.heatmap(neu_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax4, cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
ax4.set_title('Neutral - AU Correlation', fontsize=12, fontweight='bold')

# 5.5 正性情绪AU相关性热图
ax5 = plt.subplot(4, 3, 9)
pos_corr = data_dict['正性'][CORE_AU].corr()
sns.heatmap(pos_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax5, cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
ax5.set_title('Positive - AU Correlation', fontsize=12, fontweight='bold')

# 5.6 三情绪相关性差异对比
ax6 = plt.subplot(4, 2, 5)
# 计算相关系数差异
diff_corr = sad_corr - pos_corr
sns.heatmap(diff_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax6, cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
ax6.set_title('Sad vs Positive (Correlation Difference)', fontsize=12, fontweight='bold')

# 5.7 情绪时间序列对比 (以AU12为例)
ax7 = plt.subplot(4, 2, 6)
for emotion, df in data_dict.items():
    # 归一化时间 (0-100%)
    time_norm = np.linspace(0, 100, len(df))
    # 平滑曲线
    from scipy.ndimage import uniform_filter1d
    au12_smooth = uniform_filter1d(df['AU12_r'].values, size=10)
    ax7.plot(time_norm, au12_smooth, label=emotion, 
             color=EMOTION_COLORS[emotion], linewidth=2)
ax7.set_xlabel('Time (%)', fontsize=11)
ax7.set_ylabel('AU12 Intensity (Lip Corner Puller)', fontsize=11)
ax7.set_title('AU12 Time Course by Emotion', fontsize=13, fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

# 5.8 雷达图 - 情绪特征轮廓
ax8 = plt.subplot(4, 2, 7, projection='polar')
key_au = ['AU04_r', 'AU06_r', 'AU07_r', 'AU12_r', 'AU15_r', 'AU25_r']
angles = np.linspace(0, 2*np.pi, len(key_au), endpoint=False).tolist()
angles += angles[:1]  # 闭合

for emotion, df in data_dict.items():
    values = [df[au].mean() for au in key_au]
    values += values[:1]  # 闭合
    ax8.plot(angles, values, 'o-', linewidth=2, label=emotion, 
             color=EMOTION_COLORS[emotion])
    ax8.fill(angles, values, alpha=0.15, color=EMOTION_COLORS[emotion])

ax8.set_xticks(angles[:-1])
ax8.set_xticklabels([au.replace('_r', '') for au in key_au])
ax8.set_title('Emotion Profile (Radar)', fontsize=13, fontweight='bold', pad=20)
ax8.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

# 5.9 F值热图 (显著性)
ax9 = plt.subplot(4, 2, 8)
f_values = anova_df.set_index('AU')['F_statistic'].values.reshape(1, -1)
im = ax9.imshow(f_values, cmap='YlOrRd', aspect='auto')
ax9.set_xticks(range(len(CORE_AU)))
ax9.set_xticklabels([au.replace('_r', '') for au in CORE_AU], rotation=45, ha='right')
ax9.set_yticks([0])
ax9.set_yticklabels(['F-statistic'])
ax9.set_title('ANOVA F-statistic by AU', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax9)

# 添加显著性标记
for i, (_, row) in enumerate(anova_df.iterrows()):
    if row['significant_bonferroni']:
        ax9.text(i, 0, '***', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    elif row['p_value'] < 0.01:
        ax9.text(i, 0, '**', ha='center', va='center', fontsize=12, color='white', fontweight='bold')
    elif row['p_value'] < 0.05:
        ax9.text(i, 0, '*', ha='center', va='center', fontsize=12, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('three_emotion_au_analysis.png', dpi=300, bbox_inches='tight')
print("  ✅ 图表已保存: three_emotion_au_analysis.png")

# 6. 生成详细报告
print("\n" + "=" * 70)
print("📋 详细分析报告")
print("=" * 70)

report = f"""
【OpenFace AU三情绪对比分析报告】

1. 数据概况:
   - 分析对象: OpenFace 2.0提取的17个AU强度值
   - 情绪类型: 悲伤、风景(中性)、正性
   - 帧数统计:
"""
for emotion, df in data_dict.items():
    report += f"     * {emotion:4s}: {len(df):4d} 帧\n"

report += f"""
2. 核心发现:
   
   A. 最具区分度的AU (基于ANOVA F值):
"""

for _, row in anova_df.head(5).iterrows():
    sig = "***" if row['significant_bonferroni'] else "**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else "")
    report += f"      - {row['AU']:8s}: F={row['F_statistic']:7.2f}, p={row['p_value']:.4f} {sig}\n"

report += f"""
   B. 情绪特异性AU模式:
"""

# 计算每个情绪最活跃的AU
for emotion, df in data_dict.items():
    top_au = df[CORE_AU].mean().sort_values(ascending=False).head(3)
    report += f"      {emotion}情绪 Top 3 AU:\n"
    for au, val in top_au.items():
        report += f"        - {au}: {val:.3f}\n"
    report += "\n"

report += f"""
3. 统计检验说明:
   - ANOVA用于检验三组间差异
   - Bonferroni校正: α = 0.05/{len(CORE_AU)} = {0.05/len(CORE_AU):.4f}
   - 效应量: Cohen's d (Small: 0.2, Medium: 0.5, Large: 0.8)

4. 可视化输出:
   - three_emotion_au_analysis.png (综合分析图)
"""

print(report)

# 保存报告
with open('emotion_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("✅ 详细报告已保存: emotion_analysis_report.txt")

# 7. 导出统计表格
summary_table = []
for au in CORE_AU:
    row = {'AU': au}
    for emotion, df in data_dict.items():
        row[f'{emotion}_Mean'] = df[au].mean()
        row[f'{emotion}_SD'] = df[au].std()
    # 添加ANOVA结果
    anova_row = anova_df[anova_df['AU'] == au].iloc[0]
    row['F_statistic'] = anova_row['F_statistic']
    row['p_value'] = anova_row['p_value']
    summary_table.append(row)

summary_df = pd.DataFrame(summary_table)
summary_df.to_csv('au_emotion_statistics.csv', index=False)
print("✅ 统计表格已保存: au_emotion_statistics.csv")

print("\n" + "=" * 70)
print("🎉 分析完成！")
print("=" * 70)
print("输出文件:")
print("  1. three_emotion_au_analysis.png - 综合分析图表")
print("  2. emotion_analysis_report.txt - 详细分析报告")
print("  3. au_emotion_statistics.csv - 统计数据表格")
