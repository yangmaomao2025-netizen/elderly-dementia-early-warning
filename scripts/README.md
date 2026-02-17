# 分析脚本目录

**整理日期**: 2026年2月17日  
**脚本总数**: 24个  
**总大小**: ~350KB

---

## 核心分析脚本（推荐使用）

### 1. 情绪对比分析
| 脚本名称 | 功能 | 输出 |
|----------|------|------|
| `analyze_sadness_reorganized.py` | 悲伤情绪性别对比（2M+1F） | 热图、条形图、统计表 |
| `analyze_neutral_reorganized.py` | 中性情绪性别对比 | 热图、条形图、统计表 |
| `analyze_positive_reorganized.py` | 积极情绪性别对比 | 热图、条形图、统计表 |
| `cross_emotion_analysis.py` | 跨情绪综合对比（悲伤vs中性vs积极） | 综合对比热图、雷达图 |

### 2. 患者跨情绪分析
| 脚本名称 | 功能 | 输出 |
|----------|------|------|
| `analyze_female_patients_cross_emotion.py` | 女性患者F1/F2/F3跨情绪分析 | 9张相关矩阵+3张对比热图 |
| `analyze_male_patients_cross_emotion.py` | 男性患者M1/M2/M3跨情绪分析 | 9张相关矩阵+3张对比热图 |
| `analyze_patient_control_sadness_corrected.py` | 患者vs对照组悲伤对比（修正版） | 差异条形图、对比热图 |

### 3. 时间序列与动态分析
| 脚本名称 | 功能 | 输出 |
|----------|------|------|
| `au_time_series_v2.py` | AU时间序列轨迹分析 | 轨迹图、Duchenne微笑分析 |
| `calculate_facial_activity_index.py` | 面部活动性指数（AD方案核心指标） | 活动性对比图、复杂度分析 |

### 4. 相关矩阵热图
| 脚本名称 | 功能 | 输出 |
|----------|------|------|
| `generate_correlation_heatmaps.py` | 生成17×17 AU相关矩阵 | RdBu_r色图、NaN处理 |
| `generate_female_correlation_heatmaps.py` | 女性患者相关矩阵（批量） | F1/F2/F3各情绪矩阵 |
| `generate_neutral_correlation_heatmaps.py` | 中性情绪相关矩阵 | 对照组+患者组 |
| `generate_positive_correlation_heatmaps.py` | 积极情绪相关矩阵 | 对照组+患者组 |

---

## 工具脚本

| 脚本名称 | 功能 | 备注 |
|----------|------|------|
| `data_visualization.py` | 通用数据可视化 | 支持CSV/Excel输入 |
| `ad_mediation_analysis.py` | 中介效应分析 | Baron & Kenny + Bootstrap |

---

## 已弃用/旧版本脚本

以下脚本为开发过程中版本，供参考：

| 脚本名称 | 状态 | 替代脚本 |
|----------|------|----------|
| `analyze_sadness_fixed.py` | ❌ 旧版 | `analyze_sadness_reorganized.py` |
| `analyze_sadness_gender.py` | ❌ 旧版 | `analyze_sadness_reorganized.py` |
| `analyze_positive_simple.py` | ❌ 简化版 | `analyze_positive_reorganized.py` |
| `au_time_series_analysis.py` | ❌ 旧版 | `au_time_series_v2.py` |
| `cross_emotion_patient_analysis.py` | ❌ 旧版 | `cross_emotion_analysis.py` |
| `patient_control_comparison.py` | ❌ 错误版本 | `analyze_patient_control_sadness_corrected.py` |
| `three_emotion_analysis.py` | ❌ 早期版本 | `cross_emotion_analysis.py` |
| `generate_separate_plots.py` | ❌ 早期版本 | 各`analyze_*.py` |
| `generate_positive_corr_simple.py` | ❌ 简化版 | `generate_positive_correlation_heatmaps.py` |

---

## 执行顺序建议

如果是新环境首次运行，建议按以下顺序：

```bash
# 1. 基础情绪对比（对照组）
python3 analyze_sadness_reorganized.py
python3 analyze_neutral_reorganized.py
python3 analyze_positive_reorganized.py

# 2. 跨情绪综合对比
python3 cross_emotion_analysis.py

# 3. 患者分析
python3 analyze_female_patients_cross_emotion.py
python3 analyze_male_patients_cross_emotion.py
python3 analyze_patient_control_sadness_corrected.py

# 4. 高级分析（AD方案指标）
python3 calculate_facial_activity_index.py
python3 au_time_series_v2.py
```

---

## 依赖包

所有脚本依赖：
```bash
pip install pandas numpy matplotlib seaborn scipy
```

---

**最后更新**: 2026-02-17
