实验五：参数敏感性分析
目的：识别哪些参数对算法性能影响最大。
实验设置如下：
1. 全局设置
算例类型：9个确定性算例。
算例范围：D:\codex_drone - 实验五\examples\instances 中全部 9 个实例。
重复次数：每个配置、每个实例运行 1 次；本实验不计算均值与标准差，记录单次结果。
基准配置：按实例规模自动匹配。
Small 使用 config_small
Medium 使用 config_medium
Large 使用 config_large
全局日志文件：sensitivity_log.csv
每次运行记录指标：
Z_total、Z_fixed、Z_truck、Z_drone、Z_fail_expected、Z_unserved_penalty、unserved_count、sortie_count、solve_time、
sortie_feasible_rate、drone_utilization
2. 确定性场景定义
确定性场景的目的，是保证在线执行阶段客户在家状态可复现，而不是向离线规划阶段提供完全信息。
每个确定性实例必须同时包含两层信息：
规划阶段信息：客户在各时隙的到家概率向量，用于离线目标计算与启发式选择。
执行阶段信息：固定的二元真实在家状态，用于在线执行与实验复现。
固定二元真实状态必须同时包含 0 和 1，即“已知在家/不在家混合”的一般确定性场景；不得再使用“全 1”实例。
离线阶段不得直接读取执行阶段二元真值；否则会造成信息泄露，实验结论失真。
3. 实验 5.1：服务失败惩罚成本敏感性
惩罚倍数档位：1.0 / 2.0 / 3.0 / 5.0 / 10.0
修改要求：失败惩罚必须同时作用于 Z_fail_expected 与 Z_unserved_penalty。
观察重点：
Z_total 随惩罚倍数变化趋势
unserved_count 变化
sortie_count 变化
各成本分量结构变化
输出文件：
sensitivity_log.csv、exp5_1_penalty.csv、exp5_1_penalty_curve.png、exp5_1_penalty_components.png、4. 实验 5.2：无人机速度敏感性（7 档）
本实验仅以无人机速度为主实验变量。
速度档位：
0.60、0.75、0.85、1.00、1.15、1.30、1.50
载重容量 drone_capacity 保持基准不变。
电池容量 drone_battery_capacity 保持基准不变。
能耗模型要求：
飞行时间按 t = d / v 自动变化。
eta 不得保持常数；其值需随速度变化。
建议采用：
eta(v) = eta_base × (v / v_base)^2
等价地，总能耗可写成 E = eta(v) × mass × time
该实验验证的是“速度变化引起的系统响应”，因此飞行时间变化与 eta 变化必须同时保留，不可只改 v 而固定 eta。
观察重点：
Z_total 是否存在最优速度点
sortie_count
平均每个 sortie 服务客户数
sortie_feasible_rate
Z_drone
solve_time
输出文件：
exp5_2_speed_raw.csv、exp5_2_speed_summary.csv、exp5_2_speed_curve.png、exp5_2_speed_energy.png
5. 实验 5.3：无人机可服务客户比重敏感性（5 档）
本实验仅采用随机分布模式。
同一客户规模下已有三份实例版本，用户已定义其分别代表不同空间分布；因此实验脚本不再额外引入 clustered 模式。
比重档位：
20%、35%、50%、75%、100%
可服务客户比重的改变，不得通过篡改客户重量实现。
正确做法是：
保持客户原始重量不变。
单独维护“无人机可服务资格”标签或集合。
候选客户同时满足“被标记为可服务”与“实际物理约束可行”后，才允许进入无人机候选集。
若某实例在不改变原始重量分布的前提下无法达到目标比例，则该实例应判定为“需重生成”，而不是在实验脚本中强行改重量。
观察重点：
Z_total 随比重变化趋势
drone_utilization
sortie_count
Z_truck 与 Z_drone 的此消彼长关系
输出文件：
exp5_3_ratio_raw.csv、exp5_3_ratio_summary.csv、exp5_3_ratio_trend.png、exp5_3_ratio_utilization.png、
6. 汇总输出
三个子实验完成后，生成：
exp5_summary_report.html
报告内容包括：
三组实验敏感性强度对比，排序指标为 Z_total 相对基准的最大变化幅度
Small / Medium / Large 三类规模的敏感性差异分析
每个子实验 2-3 句英文结论：敏感性强度与建议取值区间
完整参数配置记录表，保证实验可复现