# Experiment 2 Data Quality Report

## 1. 文件完整性
- 识别到算例目录数: 57
- experiment2_comparison.json 读取成功: 57/57
- 缺失或解析失败文件: 无

## 2. 关键数值检查
- A_avg_cost < B_avg_cost（按规模）:
  - Small: 19/21
  - Medium: 17/18
  - Large: 18/18
- A_avg_fail < B_avg_fail（按规模）:
  - Small: 19/21
  - Medium: 18/18
  - Large: 18/18
- offline_overhead_pct 为负值的算例: 无
- A_win_rate < 50% 的算例数: 2
  - 明细:
    - small_8c_1v_seed1 (Small): 0.00%
    - small_10c_1v_seed1 (Small): 35.00%

## 3. 简要统计摘要
- delta_avg_cost_pct 均值（按规模）:
  - Small: 11.1454%
  - Medium: 14.1139%
  - Large: 11.4180%