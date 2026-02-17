#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨情绪患者分析 - 抑郁症患者的跨情绪AU模式分析
目标：识别情绪上下文无关的抑郁症标记物
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# 文件映射 - 患者数据（已确认情绪类型）
PATIENT_FILES = {
    'sadness': {
        'P1': '/root/.openclaw/media/inbound/file_24---925f9a2e-ba59-4283-829c-75d596785181.csv',
        'P2': '/root/.openclaw/media/inbound/file_25---f701a00a-5efc-44e6-8514-4510879be7a9.csv',
        'P3': '/root/.openclaw/media/inbound/file_26---2b859f5a-08e2-4713-b654-c56162c1085d.csv',
    },
    'neutral': {
        'P1': '/root/.openclaw/media/inbound/file_27---2b92310a-c235-4003-a321-a0ec90aae5ba.csv',
        'P2': '/root/.openclaw/media/inbound/file_28---a22ed529-d93d-4a5d-8bc6-2415bf19cbfe.csv',
        'P3': '/root/.openclaw/media/inbound/file_29---ee23776c-8da6-44c4-aba3-8a74eca4f174.csv',
    },
    'positive': {
        'P1': '/root/.openclaw/media/inbound/file_30---022dee2d-6990-4df3-aa41-a71b974bb24d.csv',
        'P2': '/root/.openclaw/media/inbound/file_31---b6f4948f-3181-49b5-8ea9-02070668994c.csv',
        'P3': '/root/.openclaw/media/inbound/file_32---5ceabcaa-84d6-4901-83aa-79b841590151.csv',
    }
}

# 对照组文件（用于对比）
CONTROL_FILES = {
    'sadness': {
        'M1': '/root/.openclaw/media/inbound/file_3---b3314058-964d-470d-8293-13430fdde2c6.csv',
        'M2': '/root/.openclaw/media/inbound/file_4---0dd96eb3-72ff-4ced-a1b8-c5c51fad721a.csv',
        'F1': '/root/.openclaw/media/inbound/file_5---69ad20a2-5a2f-4f18-bdef-056d8c24d515.csv',
    },
    'neutral': {
        'M1': '/root/.openclaw/media/inbound/file_18---73cb1d9c-9f3c-4f21-917a-ae9408962385.csv',
        'M2': '/root/.openclaw/media/inbound/file_19---476a6dde-2bc6-48b4-89d3-8c3e70cbd0fd.csv',
        'F1': '/root/.openclaw/media/inbound/file_20---333e020a-bdf5-44a5-b833-c3179c272ccc.csv',
    },
    'positive': {
        'M1': '/root/.openclaw/media/inbound/file_21---c1ecbaad-5700-42b7-a743-1b75f81b7ff1.csv',
        'M2': '/root/.openclaw/media/inbound/file_22---772490a5-e791-43b9-8f4a-25c2f614570a.csv',
        'F1': '/root/.openclaw/media/inbound/file_23---06535c58-c474-473b-a68d-aadcee3e3ca7.csv',
    }
}

AU_COLUMNS = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r',
              'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r',
              'AU25_r', 'AU26_r', 'AU45_r']

def load_and_clean_data(filepath):
    """加载并清洗数据"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df = df[df['success'] == 1].copy()
    return df

def extract_au_means(df):
    """提取AU均值"""
    return df[AU_COLUMNS].mean()

def calculate_cohens_d(group1, group2):
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    s1, s2 = group1.std(ddof=1), group2.std(ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std

def analyze_cross_emotion_patients():
    """分析患者跨情绪模式"""
    
    # 收集所有数据
    patient_data = {emotion: {} for emotion in PATIENT_FILES}
    control_data = {emotion: {} for emotion in CONTROL_FILES}
    
    for emotion, files in PATIENT_FILES.items():
        for subject, filepath in files.items():
            df = load_and_clean_data(filepath)
            patient_data[emotion][subject] = extract_au_means(df)
    
    for emotion, files in CONTROL_FILES.items():
        for subject, filepath in files.items():
            df = load_and_clean_data(filepath)
            control_data[emotion][subject] = extract_au_means(df)
    
    # 计算每个情绪的患者平均值
    patient_avg = {}
    for emotion in PATIENT_FILES:
        patient_avg[emotion] = pd.DataFrame(patient_data[emotion]).mean(axis=1)
    
    # 计算对照组平均值（男平均+女）
    control_avg = {}
    for emotion in CONTROL_FILES:
        m_avg = pd.DataFrame({k: v for k, v in control_data[emotion].items() if k.startswith('M')}).mean(axis=1)
        f_vals = control_data[emotion]['F1']
        control_avg[emotion] = pd.concat([m_avg, f_vals], axis=1).mean(axis=1)
    
    return patient_data, control_data, patient_avg, control_avg

def find_consistent_markers(patient_avg, control_avg):
    """识别跨情绪一致的抑郁症标记物"""
    
    emotions = ['sadness', 'neutral', 'positive']
    markers = {}
    
    for au in AU_COLUMNS:
        effects = []
        for emotion in emotions:
            p_val = patient_avg[emotion][au]
            c_val = control_avg[emotion][au]
            # 计算效应量方向
            effect = p_val - c_val
            effects.append(effect)
        
        # 检查是否在所有情绪中方向一致且幅度显著
        if all(e > 0.1 for e in effects) or all(e < -0.1 for e in effects):
            markers[au] = {
                'direction': 'elevated' if effects[0] > 0 else 'reduced',
                'emotions': {e: effects[i] for i, e in enumerate(emotions)},
                'consistency': min(abs(e) for e in effects) / max(abs(e) for e in effects)
            }
    
    return markers

def plot_patient_heatmap(patient_avg, output_dir):
    """生成患者跨情绪热力图"""
    emotions = ['sadness', 'neutral', 'positive']
    data_matrix = np.array([patient_avg[e].values for e in emotions])
    
    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=2)
    
    ax.set_xticks(range(len(AU_COLUMNS)))
    ax.set_xticklabels([au.replace('_r', '') for au in AU_COLUMNS], rotation=45, ha='right')
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(['悲伤', '中性', '积极'])
    ax.set_title('抑郁症患者跨情绪AU激活模式', fontsize=14, fontweight='bold')
    
    # 添加数值标注
    for i in range(len(emotions)):
        for j in range(len(AU_COLUMNS)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('AU激活强度', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmaps' / '患者跨情绪AU激活热力图.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_patient_vs_control_by_emotion(patient_avg, control_avg, output_dir):
    """分情绪对比患者vs对照组"""
    emotions = ['sadness', 'neutral', 'positive']
    emotion_labels = ['悲伤情绪', '中性情绪', '积极情绪']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (emotion, label) in enumerate(zip(emotions, emotion_labels)):
        ax = axes[idx]
        
        patient_vals = patient_avg[emotion].values
        control_vals = control_avg[emotion].values
        
        x = np.arange(len(AU_COLUMNS))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, patient_vals, width, label='患者', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, control_vals, width, label='对照组', color='#3498db', alpha=0.8)
        
        ax.set_xlabel('Action Units', fontsize=10)
        ax.set_ylabel('平均激活强度', fontsize=10)
        ax.set_title(f'{label}: 患者 vs 对照组', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([au.replace('_r', '') for au in AU_COLUMNS], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'barplots' / '分情绪患者对照组对比.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_consistent_markers(markers, patient_avg, control_avg, output_dir):
    """可视化一致的抑郁症标记物"""
    if not markers:
        print("未发现一致的跨情绪标记物")
        return
    
    emotions = ['sadness', 'neutral', 'positive']
    emotion_labels = ['悲伤', '中性', '积极']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (au, info) in enumerate(list(markers.items())[:6]):
        ax = axes[idx]
        
        patient_vals = [patient_avg[e][au] for e in emotions]
        control_vals = [control_avg[e][au] for e in emotions]
        
        x = np.arange(len(emotions))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, patient_vals, width, label='患者', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, control_vals, width, label='对照组', color='#3498db', alpha=0.8)
        
        ax.set_ylabel('AU激活强度', fontsize=9)
        ax.set_title(f'{au.replace("_r", "")}: {"↑" if info["direction"]=="elevated" else "↓"} 患者组\n(一致性: {info["consistency"]:.2f})', 
                    fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(emotion_labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('跨情绪一致的抑郁症标记物', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'barplots' / '跨情绪一致标记物.png', dpi=150, bbox_inches='tight')
    plt.close()

def generate_markdown_report(patient_avg, control_avg, markers, output_dir):
    """生成Markdown分析报告"""
    
    report = """# 跨情绪患者分析报告

## 分析目标
识别在悲伤、中性、积极三种情绪状态下均表现出差异的抑郁症生物标记物（情绪上下文无关标记物）

## 数据样本
- **患者组**: 3名男性抑郁症患者 (P1, P2, P3)
- **对照组**: 2名男性 + 1名女性健康被试
- **情绪条件**: 悲伤 / 中性 / 积极

## 关键发现

### 跨情绪一致的抑郁症标记物

"""
    
    if markers:
        report += f"发现 **{len(markers)}** 个跨情绪一致的标记物：\n\n"
        report += "| AU | 变化方向 | 悲伤差异 | 中性差异 | 积极差异 | 一致性指数 |\n"
        report += "|---|---|---|---|---|---|\n"
        
        for au, info in sorted(markers.items(), key=lambda x: x[1]['consistency'], reverse=True):
            direction = "↑ 升高" if info['direction'] == 'elevated' else "↓ 降低"
            emotions_data = info['emotions']
            report += f"| {au.replace('_r', '')} | {direction} | {emotions_data['sadness']:+.2f} | {emotions_data['neutral']:+.2f} | {emotions_data['positive']:+.2f} | {info['consistency']:.2f} |\n"
    else:
        report += "未发现跨情绪一致的标记物。\n\n"
    
    report += "\n### 各情绪患者 vs 对照组对比\n\n"
    
    emotions = ['sadness', 'neutral', 'positive']
    emotion_names = {'sadness': '悲伤', 'neutral': '中性', 'positive': '积极'}
    
    for emotion in emotions:
        report += f"\n#### {emotion_names[emotion]}情绪\n\n"
        report += "| AU | 患者均值 | 对照组均值 | 差异 |\n"
        report += "|---|---|---|---|\n"
        
        for au in AU_COLUMNS:
            p_val = patient_avg[emotion][au]
            c_val = control_avg[emotion][au]
            diff = p_val - c_val
            report += f"| {au.replace('_r', '')} | {p_val:.2f} | {c_val:.2f} | {diff:+.2f} |\n"
    
    report += """

## 临床意义

### 情绪上下文无关标记物
在三种情绪条件下均稳定的差异模式，可能反映：
1. **神经生物学基础改变**（如杏仁核功能异常）
2. **面部肌肉张力基线改变**（长期情绪障碍导致的肌肉模式固化）
3. **情绪调节系统功能障碍**（无法根据情境调整表情输出）

### 诊断潜力
跨情绪一致的AU模式可作为：
- **筛查工具**：无需特定情绪诱发即可检测
- **客观评估**：补充自评量表的生物学指标
- **治疗监测**：追踪干预前后的变化

---
*分析报告生成时间: 2025-02-17*
"""
    
    with open(output_dir / 'statistics' / '跨情绪患者分析报告.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

def main():
    """主函数"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(f'/root/.openclaw/workspace/analysis_results/{timestamp}_跨情绪患者分析')
    
    # 创建输出目录
    for subdir in ['heatmaps', 'barplots', 'boxplots', 'radar', 'statistics', 'raw_data']:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("跨情绪患者分析 - 识别情绪上下文无关的抑郁症标记物")
    print("="*60)
    
    # 分析数据
    print("\n[1/5] 加载数据并计算AU均值...")
    patient_data, control_data, patient_avg, control_avg = analyze_cross_emotion_patients()
    print(f"  ✓ 患者数据: 3人 × 3情绪 = 9个文件")
    print(f"  ✓ 对照组数据: 3人 × 3情绪 = 9个文件")
    
    # 识别一致标记物
    print("\n[2/5] 识别跨情绪一致的标记物...")
    markers = find_consistent_markers(patient_avg, control_avg)
    print(f"  ✓ 发现 {len(markers)} 个一致标记物")
    for au, info in markers.items():
        direction = "升高" if info['direction'] == 'elevated' else "降低"
        print(f"    - {au}: 患者组持续{direction} (一致性: {info['consistency']:.2f})")
    
    # 生成可视化
    print("\n[3/5] 生成跨情绪热力图...")
    plot_patient_heatmap(patient_avg, output_dir)
    print("  ✓ 热力图已保存")
    
    print("\n[4/5] 生成患者vs对照组分情绪对比图...")
    plot_patient_vs_control_by_emotion(patient_avg, control_avg, output_dir)
    print("  ✓ 对比图已保存")
    
    if markers:
        print("\n[5/5] 生成一致标记物可视化...")
        plot_consistent_markers(markers, patient_avg, control_avg, output_dir)
        print("  ✓ 标记物图已保存")
    
    # 生成报告
    print("\n[6/6] 生成分析报告...")
    report = generate_markdown_report(patient_avg, control_avg, markers, output_dir)
    print("  ✓ 报告已保存")
    
    # 输出汇总
    print("\n" + "="*60)
    print("分析完成!")
    print(f"结果目录: {output_dir}")
    print("="*60)
    
    return patient_avg, control_avg, markers, output_dir

if __name__ == '__main__':
    patient_avg, control_avg, markers, output_dir = main()
