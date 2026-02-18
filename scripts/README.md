# 分析脚本目录

**整理日期**: 2026年2月17日  
**脚本总数**: 15个（已清理旧版）  

---

## 核心分析脚本（推荐使用）

### 1. 情绪对比分析（对照组）
| 脚本名称 | 功能 | 输出 |
|----------|------|------|
| `analyze_sadness_reorganized.py` | 悲伤情绪性别对比（2M+1F） | 热图、条形图、统计表 |
| `analyze_neutral_reorganized.py` | 中性情绪性别对比 | 热图、条形图、统计表 |
| `analyze_positive_reorganized.py` | 积极情绪性别对比 | 热图、条形图、统计表 |

### 2. 跨情绪综合对比
| 脚本名称 | 功能 | 输出 |
|----------|------|------|
| `cross_emotion_analysis.py` | 跨情绪综合对比（悲伤vs中性vs积极） | 综合对比热图、雷达图、轨迹图 |
| `analyze_m2_three_emotions.py` | M2对照组三种情绪对比 | 热图、雷达图、差异分析 |
| `analyze_f1_three_emotions.py` | F1对照组三种情绪对比 | 热图、雷达图、差异分析 |

### 3. 患者跨情绪分析
| 脚本名称 | 功能 | 输出 |
|----------|------|------|
| `analyze_female_patients_cross_emotion.py` | 女性患者F1/F2/F3跨情绪分析 | 9张相关矩阵+3张对比热图 |
| `analyze_male_patients_cross_emotion.py` | 男性患者M1/M2/M3跨情绪分析 | 9张相关矩阵+3张对比热图 |
| `analyze_patient_control_sadness_corrected.py` | 患者vs对照组悲伤对比（修正版） | 差异条形图、对比热图 |

### 4. 时间序列与动态分析
| 脚本名称 | 功能 | 输出 |
|----------|------|------|
| `au_time_series_v2.py` | AU时间序列轨迹分析（修正版） | 轨迹图、Duchenne微笑分析 |
| `calculate_facial_activity_index.py` | 面部活动性指数（AD方案核心指标） | 活动性对比图、复杂度分析 |

### 5. 相关矩阵热图
| 脚本名称 | 功能 | 输出 |
|----------|------|------|
| `generate_correlation_heatmaps.py` | 生成17×17 AU相关矩阵 | RdBu_r色图、NaN处理 |
| `generate_female_correlation_heatmaps.py` | 女性患者相关矩阵（批量） | F1/F2/F3各情绪矩阵 |
| `generate_neutral_correlation_heatmaps.py` | 中性情绪相关矩阵 | 对照组+患者组 |
| `generate_positive_correlation_heatmaps.py` | 积极情绪相关矩阵 | 对照组+患者组 |

### 6. 工具脚本
| 脚本名称 | 功能 |
|----------|------|
| `data_visualization.py` | 通用数据可视化（支持CSV/Excel输入） |
| `ad_mediation_analysis.py` | 中介效应分析（Baron & Kenny + Bootstrap） |

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

## 数据文件映射

分析脚本使用以下文件映射（已验证）：

**对照组**:
- 悲伤: M1(file_3), M2(file_4), F1(file_5)
- 中性: M1(file_18), M2(file_19), F1(file_20)
- 积极: M1(file_21), M2(file_22), F1(file_23)

**男患者**:
- 悲伤: M1(file_27), M2(file_28), M3(file_29)
- 中性: M1(file_30), M2(file_31), M3(file_32)
- 积极: M1(file_24), M2(file_25), M3(file_26)

**女患者**:
- F1: file_33(中性-不完整), file_36(积极), file_39(悲伤)
- F2: file_34(中性), file_37(积极), file_40(悲伤)
- F3: file_35(中性), file_38(积极), file_41(悲伤)

---

**最后更新**: 2026-02-17（已清理旧版脚本）
