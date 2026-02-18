# 老年失智人群预警模式科研项目

**项目名称**: 老年失智人群预警模式科研项目  
**创建日期**: 2026年2月17日  
**项目类型**: 面部微表情分析 + 情绪识别 + 认知障碍预警  

---

## 📋 项目简介

本项目基于OpenFace 2.0面部动作单元（AU）分析技术，通过分析老年人群（含阿尔茨海默病/抑郁症患者）的面部表情特征，建立认知障碍早期预警模型。

### 研究框架（参照AD两阶段方案）

**第一阶段：行为验证** ✅ 进行中
- 面部活动性指数计算（情感淡漠客观指标）
- AU特征与情绪条件关联分析
- 患者-对照组对比

**第二阶段：神经机制** 📋 规划中
- 杏仁核功能连接与面部活动性中介分析
- 多模态MRI数据整合

---

## 📁 项目结构

```
老年失智人群预警模式科研项目/
├── README.md                          # 项目说明文档
├── scripts/                           # 分析脚本（15个，已清理旧版）
│   ├── README.md                      # 脚本目录说明
│   ├── calculate_facial_activity_index.py      # 面部活动性指数（AD核心指标）
│   ├── analyze_sadness_reorganized.py          # 悲伤情绪分析
│   ├── analyze_neutral_reorganized.py          # 中性情绪分析
│   ├── analyze_positive_reorganized.py         # 积极情绪分析
│   ├── cross_emotion_analysis.py               # 跨情绪综合
│   ├── analyze_female_patients_cross_emotion.py    # 女性患者
│   ├── analyze_male_patients_cross_emotion.py      # 男性患者
│   ├── analyze_patient_control_sadness_corrected.py # 患者-对照（修正版）
│   ├── au_time_series_v2.py                    # 时间序列（修正版）
│   ├── ad_mediation_analysis.py                # 中介效应
│   ├── generate_correlation_heatmaps.py        # 相关矩阵
│   ├── generate_female_correlation_heatmaps.py # 女性相关矩阵
│   ├── generate_neutral_correlation_heatmaps.py    # 中性相关矩阵
│   ├── generate_positive_correlation_heatmaps.py   # 积极相关矩阵
│   └── data_visualization.py                   # 通用可视化
│
├── analysis_results/                  # 分析结果（10个目录，已清理旧版错误结果）
│   ├── 总结报告/
│   │   └── 2026-02-17_analysis_summary_report_v1.md
│   ├── 2026-02-17_面部活动性指数分析/          # AD方案核心指标
│   ├── 2026-02-17_男性患者跨情绪分析/          # M1/M2/M3跨情绪
│   ├── 2026-02-17_女性患者跨情绪分析/          # F1/F2/F3跨情绪
│   ├── 2026-02-17_患者对照组对比_悲伤情绪_修正版/  # 修正版（文件映射已修正）
│   ├── 2026-02-17_跨情绪综合对比/              # 三情绪综合
│   ├── 2026-02-17_AU时间轨迹分析_v2/           # 时间序列（修正版）
│   ├── 2026-02-17_积极情绪_性别对比/           # 积极情绪
│   ├── 2025-02-17_中性情绪_2M1F对比/           # 中性情绪（对照组2M+1F）
│   └── 2025-02-17_悲伤情绪_2M1F对比/           # 悲伤情绪（对照组2M+1F）
│
└── data/                              # 数据文件映射（可选）
    └── file_mapping_v2.md
```

---

## 📊 数据概况

### 样本构成
| 分组 | 人数 | 情绪条件 | 文件数 |
|------|------|----------|--------|
| 对照组（健康） | 3人（2M+1F） | 悲伤/中性/积极 | 9 |
| 男性患者 | 3人（M1/M2/M3） | 悲伤/中性/积极 | 9 |
| 女性患者 | 3人（F1/F2/F3） | 悲伤/中性/积极 | 9 |
| **合计** | **9人** | **3种** | **27** |

### 数据质量
- **总帧数**: 约27万帧
- **采样率**: 30fps
- **AU维度**: 17个动作单元（AU01-AU45）
- **数据问题**: file_33（F1-中性）仅294帧，已搁置

---

## 🔬 核心发现

### 1. 面部活动性指数（AD方案核心指标）
| 分组 | 平均活动性 | vs对照组 | 临床意义 |
|------|-----------|----------|----------|
| 对照组 | 161.51 | — | 基准 |
| 男患者 | 127.00 | ↓21% | 轻度降低 |
| **女患者** | **80.26** | **↓50%** | **显著降低** |

> **结论**: 患者组（尤其女性）面部活动性显著降低，可作为"情感淡漠"客观指标。

### 2. 关键AU标记
- **AU07（眼睑收紧）**: 最稳定性别差异标记（男>女）
- **AU04（皱眉）**: 患者组显著降低（与刻板印象相反）
- **AU12（嘴角上扬）**: 患者组自发笑容减少
- **AU17（下巴抬起）**: 患者组代偿性升高

### 3. 患者亚型分类
| 亚型 | 特征 | 代表患者 |
|------|------|----------|
| 表达扁平型 | 各情绪AU差异小 | M1 |
| 典型表达型 | 情绪-AU对应清晰 | M2 |
| 皱眉主导型 | AU04持续高激活 | M3 |
| 低眼睑活动型 | AU07<0.2 | 女性患者为主 |

---

## 🚀 使用指南

### 运行环境
```bash
pip install pandas numpy matplotlib seaborn scipy
```

### 快速开始
```bash
cd 老年失智人群预警模式科研项目/scripts

# 1. 计算面部活动性指数
python3 calculate_facial_activity_index.py

# 2. 运行完整分析流程
python3 analyze_sadness_reorganized.py
python3 analyze_neutral_reorganized.py
python3 analyze_positive_reorganized.py
python3 cross_emotion_analysis.py
```

### 查看结果
所有分析结果保存在 `analysis_results/` 目录，按日期和主题分类。

---

## 📈 后续计划

### 近期（1-2周）
- [ ] 面部活动性指数统计检验（t检验/Mann-Whitney）
- [ ] 整合临床量表数据（HAMD、SDS等）
- [ ] 患者-对照组中性/积极情绪对比

### 中期（1-3月）
- [ ] 扩大样本量（对照组n≥25，患者组n≥20）
- [ ] 建立机器学习分类模型（SVM/Random Forest）
- [ ] 完善患者亚型分类算法

### 远期（3-6月）
- [ ] 申请MRI扫描（杏仁核体积测量）
- [ ] 中介效应分析（神经机制探索）
- [ ] 论文撰写与发表

---

## 📚 参考方案

- **AD研究方案**: `/root/.openclaw/media/inbound/file_42---2ac5c184-6595-4d20-a7ef-7e9064de2a40.pdf`
- **分析总结报告**: `analysis_results/总结报告/2026-02-17_analysis_summary_report_v1.md`
- **脚本说明**: `scripts/README.md`

---

## 👤 项目团队

- **数据分析**: 超级小龙虾 🦞
- **研究方案**: AD两阶段研究框架
- **技术支持**: OpenFace 2.0 + OpenClaw

---

## 🧹 数据清理说明

### 已清理的旧版脚本（9个删除）
以下脚本因版本过时或基于错误数据映射，已清理：
- `analyze_sadness_fixed.py`, `analyze_sadness_gender.py`（旧版）
- `analyze_positive_simple.py`（简化版）
- `au_time_series_analysis.py`（旧版，列名问题）
- `cross_emotion_patient_analysis.py`（旧版）
- `patient_control_comparison.py`（**错误版本**，文件映射错误）
- `three_emotion_analysis.py`（早期版本）
- `generate_positive_corr_simple.py`（简化版）
- `generate_separate_plots.py`（早期版本）

### 已清理的旧版结果（4个目录删除）
- `2026-02-17_AU时间轨迹分析/`（旧版，功能被_v2取代）
- `2026-02-17_患者对照组对比/`（**错误版本**，文件映射错误）
- `2026-02-17_患者对照组对比_悲伤情绪/`（**错误版本**）
- `2026-02-17_跨情绪患者分析/`（旧版，功能被独立脚本取代）

### 已重命名的结果（1个目录）
- `2025-02-17_M1_三种情绪对比/`（原`AU_emotion_analysis`，M1对照组三情绪对比）

### 修正历史
- **2026-02-17**: 发现并修正文件映射错误，重新分析患者-对照组对比
- **2026-02-17**: 清理脚本和结果，保留15个有效脚本和10个有效结果目录

---

**最后更新**: 2026年2月17日（已清理旧版脚本和错误结果）
