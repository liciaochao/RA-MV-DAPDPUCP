# Gurobi 代码审计报告（实验一）

## 0.1 现状文件清单

### 用户期望结构
- 期望目录：`codex_drone_gurobi/src/*.py`

### 实际目录（审计时）
- 实际不存在 `src/`，核心代码在：`drone.py`（2671 行）
- 算例目录实际为：`examples/instances/*.json`
- 配置目录为：`configs/config_small.json`

## 0.2 逐文件审计结果（旧代码）

### 旧代码入口
- 文件：`drone.py`
- 入口位置：`drone.py:2640`
- 读取输入：硬编码 CSV（`drone.py:2643-2646`）
- 结论：与当前 JSON 算例/配置格式不兼容

### 1) 决策变量（旧模型）
- 卡车弧变量：`X_truck_with_drone[i][j]`, `X_truck[i][j]`（`drone.py:517-519`）
- 无人机弧变量：`Y_drone[i][j]`（`drone.py:519`）
- 节点变量：`Y_launch[i]`, `Y_retrieve[i]`, `Z[i]`, `Z_truck[i]`, `Z_drone[i]`（`drone.py:455-476`）
- 时间变量：`T_truck[i]`, `T_drone[i]`, `T_service[i]`, `T_hover_s[i]`, `T_hover_r[i]`（`drone.py:458-462`）
- 能耗变量：`e[i]`, `e_fly[i][j]`, `e_service[i]`, `e_hover[i]`（`drone.py:463-465`, `520`）
- 载重变量：大量 truck/drone 到达/离开载重变量（`drone.py:466-473`）
- MTZ 序变量：`U[i]`（`drone.py:474`）

### 2) 约束（旧模型）
- 约束数量多（c1-c97 等），主要围绕：
  - 流平衡（`drone.py:542-575`）
  - 客户服务（`drone.py:576-586`）
  - depot 出发/回收（`drone.py:587-596`）
  - MTZ（`drone.py:598-614`）
  - launch/retrieve、时序、时间窗、载重、能耗等（`drone.py:615-1567`）
- 主要问题：
  - 与实验一目标不一致（见下一节）
  - 变量语义复杂且混杂 pickup/delivery/在线逻辑，验证困难
  - 输入和参数来源不符合当前 JSON+config 规范

### 3) 目标函数（旧模型）
- 位置：`drone.py:522-540`
- 当前包含：
  - truck 路径成本
  - drone 能耗成本
  - fixed cost
  - **未服务惩罚项**：`(1 - Z[i]) * perpenalize`（`drone.py:536-539`）
- 结论：与实验一要求“仅 Z_fixed + Z_truck + Z_drone”不一致

### 4) sortie 建模（旧模型）
- 旧模型通过 `Y_drone[i][j]` 弧变量建模，不是清晰的“sortie 集合覆盖”结构
- 未按实验一要求明确“多 sortie（每次最多 3 客户）”结构化表达

### 5) 算例读取（旧代码 vs 现有 JSON）
- 旧代码读取 CSV：`read_data(path_customer, path_caf, customer_num)`（`drone.py:56`）
- 当前算例为 JSON：`examples/instances/*.json`
- 差异：输入字段完全不同，旧代码不能直接读取 JSON

### 6) 重规划代码位置
- `RollingController` 整段在 `drone.py` 中注释保留（约 `drone.py:2142` 之后）
- 已在其前添加注释：
  - `# 实验一（确定性环境 p=1）不使用此模块，保留供后续实验使用`（`drone.py:2141`）

### 7) “求解异常快”可能原因
- 旧入口根本未读取当前 JSON 算例，可能求解了错误/极小实例
- 目标含惩罚项，可能把服务逻辑扭曲
- 参数硬编码与当前 config 不一致
- 约束体系与实验一目标不匹配，导致模型实际难度偏低或行为失真
- 旧代码大量变量/约束但缺乏面向当前数据的有效绑定

## 0.3 JSON 结构确认

- 算例顶层 keys：
  - `depot_id`, `vehicle_pair_count`, `depot`, `truck_distance_scale`, `drone_distance_scale`, `trucks`, `drones`, `delivery_customers`, `pickup_customers`, `customers`
- 配置顶层 keys：
  - `time`, `vehicle`, `energy`, `cost`, `beta_schedule`, `constraints`, `aco`, `alns`, `etprc`, `online_alns`, `monte_carlo`

## 建模策略选择

- 采用策略 A：预枚举合法 sortie 候选 + MILP 选择
- 出于规模控制，采用“预枚举后筛选入模候选上限（默认 400）”
- 理由：避免在主 MILP 中引入 sortie 内部排序变量，保持模型可解
