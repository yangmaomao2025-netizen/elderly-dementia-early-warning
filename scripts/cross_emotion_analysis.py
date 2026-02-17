#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨情绪AU数据综合对比分析
对比三种情绪：悲伤、中性、积极
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# AU列表（17个）
AU_LIST = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 
           'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 
           'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']

# 创建输出目录
timestamp = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = f'/root/.openclaw/workspace/analysis_results/{timestamp}_跨情绪综合对比'

# 如果目录已存在，删除旧文件
if os.path.exists(OUTPUT_DIR):
    import shutil
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for subdir in ['cross_emotion_charts', 'au_trajectory', 'gender_heatmaps', 'statistics']:
    os.makedirs(f'{OUTPUT_DIR}/{subdir}', exist_ok=True)

print("=" * 70)
print("跨情绪AU数据综合对比分析")
print("=" * 70)

# ==================== 1. 手动整理三种情绪数据 ====================
print("\n【1. 整理三种情绪数据】")

# 悲伤情绪数据
sadness_data = {
    'AU': AU_LIST,
    'Male_Mean': [0.1513, 0.0550, 1.3823, 0.0520, 0.3895, 1.6430, 0.0422, 0.3932, 0.4701, 0.0035, 0.0914, 0.3249, 0.0633, 0.1016, 0.3253, 0.2632, 0.2835],
    'Female_Mean': [0.0780, 0.0405, 1.0695, 0.0374, 0.0001, 0.0358, 0.0283, 0.5557, 0.1877, 0.0000, 0.0635, 0.1751, 0.0443, 0.0657, 0.1120, 0.1915, 0.1965],
    'Gender_Diff': [0.0733, 0.0145, 0.3128, 0.0146, 0.3894, 1.6072, 0.0139, -0.1625, 0.2823, 0.0035, 0.0279, 0.1498, 0.0190, 0.0359, 0.2133, 0.0717, 0.0869]
}

# 中性情绪数据
neutral_data = {
    'AU': AU_LIST,
    'Male_Mean': [0.2485, 0.1840, 1.1723, 0.0490, 0.4017, 1.3560, 0.1043, 0.9283, 0.2540, 0.2237, 0.0558, 0.1675, 0.0343, 0.1795, 0.3020, 0.2683, 0.2227],
    'Female_Mean': [0.0190, 0.0120, 0.8230, 0.0150, 0.0000, 0.0180, 0.0230, 0.4310, 0.0500, 0.0000, 0.0230, 0.0360, 0.0200, 0.0370, 0.0410, 0.0900, 0.0840],
    'Gender_Diff': [0.2295, 0.1720, 0.3493, 0.0340, 0.4017, 1.3380, 0.0813, 0.4973, 0.2040, 0.2237, 0.0328, 0.1315, 0.0143, 0.1425, 0.2610, 0.1783, 0.1387]
}

# 积极情绪数据 (从analysis_report.txt提取)
positive_data = {
    'AU': AU_LIST,
    'Male_Mean': [0.191, 0.128, 1.518, 0.037, 0.550, 1.509, 0.090, 0.967, 1.222, 0.134, 0.117, 0.562, 0.077, 0.134, 0.626, 0.255, 0.202],
    'Female_Mean': [0.114, 0.053, 0.507, 0.035, 0.085, 0.525, 0.041, 0.819, 0.755, 0.050, 0.079, 0.192, 0.090, 0.075, 0.179, 0.125, 0.126],
    'Gender_Diff': [0.078, 0.075, 1.010, 0.002, 0.465, 0.984, 0.049, 0.148, 0.467, 0.083, 0.037, 0.369, -0.013, 0.060, 0.447, 0.130, 0.076]
}

EMOTIONS = {
    '悲伤': {'df': pd.DataFrame(sadness_data), 'color': '#3498db'},
    '中性': {'df': pd.DataFrame(neutral_data), 'color': '#95a5a6'},
    '积极': {'df': pd.DataFrame(positive_data), 'color': '#e74c3c'}
}

for emotion, info in EMOTIONS.items():
    print(f"  ✓ {emotion}: {len(info['df'])}个AU指标已加载")

# ==================== 2. 构建跨情绪对比表 ====================
print("\n【2. 构建跨情绪性别差异对比】")

cross_emotion_data = []
for emotion, info in EMOTIONS.items():
    df = info['df']
    df['Emotion'] = emotion
    df['Male_GT_Female'] = df['Gender_Diff'] > 0
    cross_emotion_data.append(df[['Emotion', 'AU', 'Male_Mean', 'Female_Mean', 'Gender_Diff', 'Male_GT_Female']])

cross_df = pd.concat(cross_emotion_data, ignore_index=True)
cross_pivot = cross_df.pivot(index='AU', columns='Emotion', values='Gender_Diff')
cross_pivot = cross_pivot[['悲伤', '中性', '积极']]  # 固定顺序

print("跨情绪性别差异矩阵:")
print(cross_pivot.round(3))

# 保存CSV
cross_pivot.to_csv(f'{OUTPUT_DIR}/statistics/cross_emotion_gender_diff.csv')

# ==================== 3. 跨情绪热力图 ====================
print("\n【3. 生成跨情绪性别差异热力图】")

fig, ax = plt.subplots(figsize=(8, 12))

# 使用diverging colormap，中心为0
sns.heatmap(cross_pivot, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, vmin=-0.5, vmax=1.8, linewidths=0.5, ax=ax)
ax.set_title('跨情绪性别差异热力图\n(男性均值 - 女性均值)', fontsize=14, fontweight='bold')
ax.set_xlabel('情绪类型', fontsize=12)
ax.set_ylabel('Action Unit', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/gender_heatmaps/cross_emotion_gender_diff_heatmap.png', 
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 跨情绪热力图已保存")

# ==================== 4. 关键AU跨情绪轨迹图 ====================
print("\n【4. 生成关键AU跨情绪轨迹图】")

# 选择差异最大的几个AU
key_aus = ['AU04_r', 'AU07_r', 'AU06_r', 'AU12_r']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, au in enumerate(key_aus):
    ax = axes[idx]
    
    au_data = cross_df[cross_df['AU'] == au].set_index('Emotion')
    au_data = au_data.reindex(['悲伤', '中性', '积极'])
    
    x = range(len(au_data))
    width = 0.35
    
    male_vals = au_data['Male_Mean'].values
    female_vals = au_data['Female_Mean'].values
    
    bars1 = ax.bar([i - width/2 for i in x], male_vals, width, label='男性', color='#3498db')
    bars2 = ax.bar([i + width/2 for i in x], female_vals, width, label='女性', color='#e91e63')
    
    ax.set_xlabel('情绪类型', fontsize=11)
    ax.set_ylabel(f'{au} 激活强度', fontsize=11)
    ax.set_title(f'{au} 跨情绪激活对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(au_data.index)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/au_trajectory/key_au_cross_emotion_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 关键AU跨情绪对比图已保存")

# ==================== 5. AU07深度分析 ====================
print("\n【5. AU07 (眼睑收紧) 深度分析】")

au07_data = cross_df[cross_df['AU'] == 'AU07_r'].set_index('Emotion')
au07_data = au07_data.reindex(['悲伤', '中性', '积极'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 左图：AU07激活强度对比
x = range(len(au07_data))
width = 0.35
bars1 = ax1.bar([i - width/2 for i in x], au07_data['Male_Mean'], width, label='男性', color='#3498db')
bars2 = ax1.bar([i + width/2 for i in x], au07_data['Female_Mean'], width, label='女性', color='#e91e63')
ax1.set_xlabel('情绪类型', fontsize=12)
ax1.set_ylabel('AU07 激活强度', fontsize=12)
ax1.set_title('AU07 (眼睑收紧) 跨情绪激活强度', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(au07_data.index)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
               xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')

# 右图：AU07性别差异
ax2.plot(x, au07_data['Gender_Diff'], marker='o', linewidth=2.5, markersize=12, color='#e74c3c')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('情绪类型', fontsize=12)
ax2.set_ylabel('性别差异 (男性-女性)', fontsize=12)
ax2.set_title('AU07 性别差异变化趋势', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(au07_data.index)
ax2.grid(alpha=0.3)

# 添加差异值标签
for i, diff in enumerate(au07_data['Gender_Diff']):
    ax2.annotate(f'{diff:.2f}', (i, diff), textcoords="offset points", 
                xytext=(0, 10), ha='center', fontsize=12, fontweight='bold', color='#e74c3c')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/au_trajectory/AU07_deep_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"  AU07 跨情绪性别差异:")
for emotion in au07_data.index:
    diff = au07_data.loc[emotion, 'Gender_Diff']
    male = au07_data.loc[emotion, 'Male_Mean']
    female = au07_data.loc[emotion, 'Female_Mean']
    print(f"    {emotion}: 男={male:.2f}, 女={female:.2f}, 差异={diff:.2f}")
print("  ✓ AU07深度分析图已保存")

# ==================== 6. 性别差异一致性分析 ====================
print("\n【6. 性别差异一致性分析】")

# 统计男性>女性的AU数量
consistency_data = []
for au in AU_LIST:
    au_data = cross_df[cross_df['AU'] == au].set_index('Emotion')
    au_data = au_data.reindex(['悲伤', '中性', '积极'])
    if len(au_data) == 3:
        male_gt_count = au_data['Male_GT_Female'].sum()
        consistency_data.append({
            'AU': au,
            'Male_GT_Count': int(male_gt_count),
            'Pattern': '男性>女性 (一致)' if male_gt_count == 3 else 
                      '女性>男性 (一致)' if male_gt_count == 0 else
                      '混合模式'
        })

consistency_df = pd.DataFrame(consistency_data)
print(consistency_df.to_string(index=False))

# 可视化
fig, ax = plt.subplots(figsize=(10, 6))
pattern_counts = consistency_df['Pattern'].value_counts()
colors = {'男性>女性 (一致)': '#3498db', '混合模式': '#f39c12', '女性>男性 (一致)': '#e91e63'}
bar_colors = [colors.get(p, '#95a5a6') for p in pattern_counts.index]

bars = ax.barh(pattern_counts.index, pattern_counts.values, color=bar_colors)
ax.set_xlabel('AU数量', fontsize=12)
ax.set_title('性别差异一致性分布\n(3种情绪中男性>女性的AU数量)', fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, pattern_counts.values):
    ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, str(val), 
            va='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cross_emotion_charts/gender_consistency_pattern.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 性别差异一致性图已保存")

# ==================== 7. 生成对比雷达图 ====================
print("\n【7. 生成跨情绪雷达图】")

fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))

emotions_list = ['悲伤', '中性', '积极']
selected_aus = ['AU01_r', 'AU02_r', 'AU04_r', 'AU06_r', 'AU07_r', 'AU10_r', 'AU12_r', 'AU25_r']
angles = np.linspace(0, 2*np.pi, len(selected_aus), endpoint=False).tolist()
angles += angles[:1]

for idx, (emotion, ax) in enumerate(zip(emotions_list, axes)):
    emotion_data = cross_df[cross_df['Emotion'] == emotion]
    emotion_data = emotion_data.set_index('AU')
    
    male_vals = [emotion_data.loc[au, 'Male_Mean'] if au in emotion_data.index else 0 for au in selected_aus]
    female_vals = [emotion_data.loc[au, 'Female_Mean'] if au in emotion_data.index else 0 for au in selected_aus]
    
    male_vals += male_vals[:1]
    female_vals += female_vals[:1]
    
    ax.plot(angles, male_vals, 'o-', linewidth=2, label='男性', color='#3498db')
    ax.fill(angles, male_vals, alpha=0.25, color='#3498db')
    ax.plot(angles, female_vals, 'o-', linewidth=2, label='女性', color='#e91e63')
    ax.fill(angles, female_vals, alpha=0.25, color='#e91e63')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([au.replace('_r', '') for au in selected_aus], fontsize=9)
    ax.set_title(f'{emotion}情绪', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cross_emotion_charts/radar_comparison_all_emotions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ 跨情绪雷达图已保存")

# ==================== 8. AU激活模式分组对比 ====================
print("\n【8. 生成AU分组激活模式图】")

# 按功能分组AU
au_groups = {
    '眉毛区域': ['AU01_r', 'AU02_r', 'AU04_r'],
    '眼睑区域': ['AU05_r', 'AU07_r'],
    '脸颊/嘴部': ['AU06_r', 'AU10_r', 'AU12_r', 'AU14_r'],
    '其他': ['AU09_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r']
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (group_name, au_list) in enumerate(au_groups.items()):
    ax = axes[idx]
    
    # 准备数据
    group_data = []
    for emotion in ['悲伤', '中性', '积极']:
        emotion_data = cross_df[cross_df['Emotion'] == emotion]
        emotion_data = emotion_data.set_index('AU')
        for au in au_list:
            if au in emotion_data.index:
                group_data.append({
                    'Emotion': emotion,
                    'AU': au.replace('_r', ''),
                    'Male': emotion_data.loc[au, 'Male_Mean'],
                    'Female': emotion_data.loc[au, 'Female_Mean']
                })
    
    group_df = pd.DataFrame(group_data)
    
    # 绘制分组柱状图
    x = np.arange(len(group_df))
    width = 0.35
    
    ax.bar(x - width/2, group_df['Male'], width, label='男性', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, group_df['Female'], width, label='女性', color='#e91e63', alpha=0.8)
    
    ax.set_xlabel('AU', fontsize=11)
    ax.set_ylabel('激活强度', fontsize=11)
    ax.set_title(f'{group_name} AU激活对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['Emotion']}\n{row['AU']}" for _, row in group_df.iterrows()], fontsize=9)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cross_emotion_charts/au_group_activation_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ AU分组激活模式图已保存")

# ==================== 9. 综合报告 ====================
print("\n【9. 生成综合分析报告】")

report = f"""
================================================================================
跨情绪AU数据综合对比分析报告
================================================================================
分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
情绪类型: 悲伤、中性、积极 (3种)
被试构成: 每种情绪 2男 + 1女

================================================================================
一、总体发现
================================================================================

【性别差异一致性】
• 男性>女性 (3种情绪一致): {len(consistency_df[consistency_df['Pattern'] == '男性>女性 (一致)'])} 个AU
• 女性>男性 (3种情绪一致): {len(consistency_df[consistency_df['Pattern'] == '女性>男性 (一致)'])} 个AU
• 混合模式 (情绪依赖): {len(consistency_df[consistency_df['Pattern'] == '混合模式'])} 个AU

男性>女性的AU列表:
{', '.join(consistency_df[consistency_df['Pattern'] == '男性>女性 (一致)']['AU'].tolist())}

混合模式AU列表:
{', '.join(consistency_df[consistency_df['Pattern'] == '混合模式']['AU'].tolist())}

【关键AU跨情绪表现】

1. AU07 (眼睑收紧) - "性别差异之王"
   悲伤: 男=1.64, 女=0.04, 差异=+1.60
   中性: 男=1.36, 女=0.02, 差异=+1.34
   积极: 男=1.51, 女=0.53, 差异=+0.98
   → 在所有情绪中男性激活均显著高于女性
   → 悲伤情绪中差异最大，积极情绪中差异相对较小

2. AU04 (眉毛下垂) 
   悲伤: 男=1.38, 女=1.07, 差异=+0.31
   中性: 男=1.17, 女=0.82, 差异=+0.35
   积极: 男=1.52, 女=0.51, 差异=+1.01
   → 积极情绪中性别差异最大（意外发现）

3. AU12 (嘴角上扬/微笑标志)
   悲伤: 男=0.47, 女=0.19, 差异=+0.28
   中性: 男=0.25, 女=0.05, 差异=+0.20
   积极: 男=1.22, 女=0.76, 差异=+0.47
   → 积极情绪激活最强，男性微笑幅度更大

4. AU06 (脸颊提升/杜氏微笑标志)
   悲伤: 男=0.39, 女=0.00, 差异=+0.39
   中性: 男=0.40, 女=0.00, 差异=+0.40
   积极: 男=0.55, 女=0.09, 差异=+0.47
   → 积极情绪中激活最强，女性在负性/中性情绪中几乎无激活

================================================================================
二、情绪特异性发现
================================================================================

【悲伤情绪】
• 最大性别差异: AU07 (眼睑收紧, diff=1.60)
• 次要差异: AU06 (脸颊提升, diff=0.39)
• 特点: 男性在悲伤时眼睑收紧和眉毛活动更明显，女性表达更内敛

【中性情绪】
• 最大性别差异: AU07 (眼睑收紧, diff=1.34)
• 次要差异: AU06 (脸颊提升, diff=0.40)
• 特点: AU06和AU14在女性中完全无激活

【积极情绪】
• 最大性别差异: AU04 (眉毛下垂, diff=1.01) - 意外发现
• 次要差异: AU12 (嘴角上扬, diff=0.47)
• 特点: 
  - 男性在表达快乐时面部整体激活更强
  - AU04在积极情绪中的差异远超悲伤和中性（需注意：AU04通常与皱眉相关）
  - AU12和AU06同时激活，显示典型的"杜氏微笑"模式

================================================================================
三、跨情绪模式总结
================================================================================

【稳定的性别差异模式】
1. AU07 (眼睑收紧): 跨情绪最稳定的性别差异指标
   - 在所有3种情绪中均显示男性>女性
   - 可能反映男性面部肌肉张力普遍更高
   - 悲伤和中性情绪中差异最大

2. AU06 (脸颊提升): 
   - 男性在3种情绪中均有激活，女性几乎无激活
   - 与AU12协同形成微笑表情

3. AU04 (眉毛下垂):
   - 男性激活普遍高于女性
   - 积极情绪中差异意外增大

【情绪依赖的性别差异模式】
1. AU10 (上唇提升):
   - 悲伤: 女性 > 男性 (反向)
   - 中性/积极: 男性 > 女性
   - 可能反映悲伤时女性更多表达厌恶/轻蔑相关表情

================================================================================
四、研究启示
================================================================================

1. AU07作为性别差异生物标记的潜力
   - 在所有情绪条件下均稳定显示男性>女性
   - 建议未来研究将AU07作为性别分层的协变量

2. 积极情绪的性别表达模式
   - 传统观念认为女性更善于表达积极情绪
   - 但本数据显示男性在AU12(微笑)和AU06(脸颊提升)上激活更强
   - 可能的解释：男性在实验情境下更努力表达"标准"快乐表情

3. 中性情绪作为基线的重要性
   - 能排除情绪内容影响，反映纯性别差异
   - AU06和AU14的性别特异性在中性情绪中最明显

4. AU04在积极情绪中的意外表现
   - 需检查数据：AU04通常与皱眉相关
   - 可能解释：积极视频中包含幽默/意外元素引发短暂皱眉

================================================================================
五、建议后续分析
================================================================================

1. 验证AU04在积极情绪中的激活原因
2. 增加样本量（当前2M+1F，统计效力有限）
3. 考虑个体差异（M1 vs M2的内部变异）
4. 时间序列分析：观察AU激活的动态变化模式

================================================================================
报告结束
================================================================================
"""

with open(f'{OUTPUT_DIR}/statistics/cross_emotion_comprehensive_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)

print("\n" + "=" * 70)
print("跨情绪综合对比分析完成！")
print(f"输出目录: {OUTPUT_DIR}")
print("\n生成的文件:")
print("  📊 cross_emotion_charts/ - 跨情绪对比图表")
print("  📈 au_trajectory/ - AU轨迹分析图")
print("  🗺️  gender_heatmaps/ - 性别差异热力图")
print("  📋 statistics/ - 统计分析报告")
print("=" * 70)
