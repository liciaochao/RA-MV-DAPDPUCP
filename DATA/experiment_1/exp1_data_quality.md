# exp1 Data Quality Report

## 1. 文件完整性检查

- objective_summary.json: 171/171
- timing.json: 171/171
- convergence_data.json: 171/171

缺失或解析失败文件：
- 无

## 2. 数值异常检查

- obj_final <= 0 记录数: 0
- time_s == 0 或 > 3600 记录数: 27
  - medium_40c_3v_seed2
  - medium_50c_3v_seed1
  - medium_60c_4v_seed2
  - medium_80c_5v_seed1
  - medium_80c_5v_seed2
  - medium_80c_5v_seed3
  - medium_100c_7v_seed1
  - medium_100c_7v_seed2
  - medium_100c_7v_seed3
  - large_150c_9v_seed1
  - large_150c_9v_seed2
  - large_150c_9v_seed3
  - large_180c_11v_seed1
  - large_180c_11v_seed2
  - large_180c_11v_seed3
  - large_200c_13v_seed1
  - large_200c_13v_seed2
  - large_200c_13v_seed3
  - large_250c_15v_seed1
  - large_250c_15v_seed2
  - large_250c_15v_seed3
  - large_280c_18v_seed1
  - large_280c_18v_seed2
  - large_280c_18v_seed3
  - large_300c_20v_seed1
  - large_300c_20v_seed2
  - large_300c_20v_seed3
- |delta| > 20% 记录数: 27
  - small_18c_2v_seed1 | delta_vs_ACO=42.09489846628532 | delta_vs_ALNS=42.09489846628532
  - small_18c_2v_seed2 | delta_vs_ACO=40.19677561424152 | delta_vs_ALNS=38.69575647436209
  - small_18c_2v_seed3 | delta_vs_ACO=43.08950691396013 | delta_vs_ALNS=0.06835077527954372
  - small_20c_2v_seed1 | delta_vs_ACO=40.93528172552566 | delta_vs_ALNS=39.0047294786736
  - small_20c_2v_seed2 | delta_vs_ACO=43.14488247484453 | delta_vs_ALNS=41.467684654377074
  - small_20c_2v_seed3 | delta_vs_ACO=39.748148610291764 | delta_vs_ALNS=39.004404903602484
  - medium_30c_2v_seed1 | delta_vs_ACO=37.46227028735302 | delta_vs_ALNS=35.23439855936248
  - medium_30c_2v_seed2 | delta_vs_ACO=37.23818457903545 | delta_vs_ALNS=33.832970977958524
  - medium_40c_3v_seed1 | delta_vs_ACO=29.686419753083516 | delta_vs_ALNS=27.693207011859787
  - medium_40c_3v_seed2 | delta_vs_ACO=28.354648474107293 | delta_vs_ALNS=2.0501079774832927
  - medium_40c_3v_seed3 | delta_vs_ACO=29.73792446257486 | delta_vs_ALNS=1.8426998473875424
  - medium_50c_3v_seed1 | delta_vs_ACO=26.745592984136536 | delta_vs_ALNS=24.80726374535139
  - medium_50c_3v_seed2 | delta_vs_ACO=22.687857946756765 | delta_vs_ALNS=22.340399339680015
  - medium_60c_4v_seed1 | delta_vs_ACO=23.184868509146586 | delta_vs_ALNS=2.014229268148186
  - medium_60c_4v_seed2 | delta_vs_ACO=40.15199448757059 | delta_vs_ALNS=36.37808810265138
  - medium_60c_4v_seed3 | delta_vs_ACO=22.14751280101172 | delta_vs_ALNS=1.9252895924057707
  - medium_80c_5v_seed1 | delta_vs_ACO=32.34403207233994 | delta_vs_ALNS=19.32593940182516
  - medium_80c_5v_seed3 | delta_vs_ACO=20.0051583844624 | delta_vs_ALNS=17.29221218356348
  - medium_100c_7v_seed1 | delta_vs_ACO=36.63497589453154 | delta_vs_ALNS=34.92715285303978
  - medium_100c_7v_seed2 | delta_vs_ACO=34.011546084956834 | delta_vs_ALNS=12.121256228537773
  - medium_100c_7v_seed3 | delta_vs_ACO=26.765469923892315 | delta_vs_ALNS=1.5564538801682355
  - large_150c_9v_seed1 | delta_vs_ACO=33.34410539348758 | delta_vs_ALNS=20.258074393524232
  - large_150c_9v_seed2 | delta_vs_ACO=20.1238456690708 | delta_vs_ALNS=18.40284588734668
  - large_150c_9v_seed3 | delta_vs_ACO=30.529356893773212 | delta_vs_ALNS=25.354708206516275
  - large_180c_11v_seed1 | delta_vs_ACO=23.525736432802756 | delta_vs_ALNS=7.252957865732071
  - large_180c_11v_seed3 | delta_vs_ACO=23.398609393854755 | delta_vs_ALNS=21.192231982376335
  - large_200c_13v_seed3 | delta_vs_ACO=28.472455381161687 | delta_vs_ALNS=16.16163475641471
- 三算法目标值完全相同 记录数: 1
  - small_8c_1v_seed1 | obj=29.294488590261764

## 3. 字段名称映射记录

### objective_summary.json
- algo=ACO_ALNS, final=optimized_objective, initial=initial_objective, count=57
- algo=Pure_ACO, final=optimized_objective, initial=initial_objective, count=57
- algo=Pure_ALNS, final=optimized_objective, initial=initial_objective, count=57
### timing.json
- algo=ACO_ALNS, total=total_seconds, count=57
- algo=Pure_ACO, total=total_seconds, count=57
- algo=Pure_ALNS, total=total_seconds, count=57
### convergence_data.json
- algo=ACO_ALNS, iter=aco_iteration, obj=best_cost, pheromone_keys=, count=57
- algo=Pure_ACO, iter=global_step, obj=best_objective, pheromone_keys=, count=57
- algo=Pure_ALNS, iter=iteration, obj=best_objective, pheromone_keys=, count=57

## 4. 简要统计摘要

### 三算法在Small/Medium/Large上的平均目标值
- size=Small, algo=ACO_ALNS, obj_mean=34.33076290444862
- size=Small, algo=Pure_ACO, obj_mean=42.3509774008798
- size=Small, algo=Pure_ALNS, obj_mean=40.5158738455389
- size=Medium, algo=ACO_ALNS, obj_mean=121.10400546746352
- size=Medium, algo=Pure_ACO, obj_mean=167.44197880112478
- size=Medium, algo=Pure_ALNS, obj_mean=144.32231842767044
- size=Large, algo=ACO_ALNS, obj_mean=689.841960188622
- size=Large, algo=Pure_ACO, obj_mean=823.4713110018633
- size=Large, algo=Pure_ALNS, obj_mean=738.6273932140913
### ACO-ALNS相对两基准的平均改进率
- vs Pure_ACO: 18.80218256824939
- vs Pure_ALNS: 10.822329170307619