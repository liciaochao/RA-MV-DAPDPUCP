from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
import zipfile


MARKDOWN = """# 任务二：离线优化算法代码分析（step2_offline_optimizationv3）

本文档严格基于 `D:\\codex_drone` 当前代码实现生成，仅聚焦离线优化阶段（Offline）。

- 代码扫描范围：`src/spd/*.py`、`tests/*.py`、`examples/configs/*.json`、`run_experiments.py`
- 核心离线模块：`src/spd/aco_alns.py`、`src/spd/etprc.py`、`src/spd/core.py`、`src/spd/config.py`、`src/spd/main.py`
- 关键事实：当前离线主算法是 `ACO + ALNS` 混合元启发式；同时保留 `PureACO` 消融分支

## 1. 优化目标与优化策略总述

### 1.1 优化目标（Objective）

离线目标函数在 `src/spd/core.py:729` (`compute_objective`) 定义，目标最小化：

`Z = Z_fixed + Z_truck + Z_drone + Z_fail_expected + beta * Z_tw_penalty + Z_unserved_penalty`

- `Z_fixed`：启用车辆对固定成本（`src/spd/core.py:745`）
- `Z_truck`：卡车行驶成本（`src/spd/core.py:755`）
- `Z_drone`：无人机能耗成本（`src/spd/core.py:761`，能耗模型见 `src/spd/core.py:380`）
- `Z_fail_expected`：基于到达时隙在家概率的期望失败成本（`src/spd/core.py:704`、`src/spd/core.py:771`）
- `Z_tw_penalty`：时间窗迟到软惩罚（`src/spd/core.py:766` 到 `src/spd/core.py:773`）
- `Z_unserved_penalty`：未服务客户惩罚，防止“丢客户”伪改进（`src/spd/core.py:774` 到 `src/spd/core.py:779`）

### 1.2 优化策略（Algorithmic Strategy）

当前代码采用“三层结构”：

- 初始化构造层：`ETPRCBuilder.build_initial_solution` 生成可行初解（`src/spd/etprc.py:25`）
- 外层全局搜索层（ACO）：多轮迭代 + 多蚂蚁构造 + 信息素更新（`src/spd/aco_alns.py:1581`）
- 内层局部搜索层（ALNS）：破坏-修复 + SA 接受准则 + 自适应权重（`src/spd/aco_alns.py:997`）

对应算法类型：

- 元启发式：`ACO`、`ALNS`、`SA`
- 局部搜索：`ALNS` 的 destroy/repair 邻域
- 进化算法：当前代码中未实现 GA/EA 类操作（无交叉/变异种群框架）

## 2. 优化主循环逻辑（函数名 + 文件名）

### 2.1 离线入口与求解器选择

- CLI 入口：`src/spd/main.py:221` (`main`)
- 求解器构建：`src/spd/main.py:190` (`_build_offline_solver`)
- 默认离线求解器：`ACOALNSSolver(etprc_builder=ETPRCBuilder(), alns_optimizer=ALNSOptimizer(...))`（`src/spd/main.py:200`）

### 2.2 ACO 外层主循环

主循环函数：`src/spd/aco_alns.py:1581` (`ACOALNSSolver.solve`)

每轮 ACO 迭代流程：

- 生成每只蚂蚁参数（含信息素矩阵、beta、seed）：`src/spd/aco_alns.py:1678`
- 每只蚂蚁执行“构造 + ALNS”：`_run_single_ant`（`src/spd/aco_alns.py:1315`）
- 聚合本轮最优解、更新全局最优：`src/spd/aco_alns.py:1738` 到 `src/spd/aco_alns.py:1751`
- 信息素蒸发与沉积：`update_pheromone`（`src/spd/aco_alns.py:1999`）
- 动态 beta 更新：`advance_beta_schedule`（`src/spd/aco_alns.py:2049`）
- 终止判定：`no_improve_count >= no_improve_max`（`src/spd/aco_alns.py:1767`）

### 2.3 ALNS 内层主循环

主循环函数：`src/spd/aco_alns.py:997` (`ALNSOptimizer.optimize`)

每次 ALNS 迭代流程：

- 轮盘赌选择 destroy 与 repair 算子（`src/spd/aco_alns.py:1053` 到 `src/spd/aco_alns.py:1061`）
- 随机确定 remove_count（`src/spd/aco_alns.py:1056` 到 `src/spd/aco_alns.py:1058`）
- 执行 destroy + repair（`src/spd/aco_alns.py:1065` 到 `src/spd/aco_alns.py:1080`）
- 若出现未服务客户或结构损坏，则 fallback 重建（`src/spd/aco_alns.py:1082` 到 `src/spd/aco_alns.py:1091`）
- 周期触发跨车辆对操作（`src/spd/aco_alns.py:1103` 到 `src/spd/aco_alns.py:1109`）
- 结构与物理约束检查（跳过硬时间窗）`_structural_and_physical_check`（`src/spd/aco_alns.py:844`）
- 计算候选目标值（带当前 beta）（`src/spd/aco_alns.py:1124` 到 `src/spd/aco_alns.py:1130`）
- SA 接受/拒绝 + 奖励打分（`src/spd/aco_alns.py:1184` 到 `src/spd/aco_alns.py:1201`）
- 更新当前解、可行最优解、算子权重、温度、历史（`src/spd/aco_alns.py:1202` 到 `src/spd/aco_alns.py:1285`）

## 3. 邻域结构 / 操作算子详细描述

### 3.1 Destroy 算子（D）

#### D1: 随机移除 `d1_random_removal`

- 操作定义：从当前解的全部已分配客户中随机抽取 `k` 个并移除，进入未服务集合（`src/spd/aco_alns.py:155`）
- 触发条件：在 ALNS 迭代中被轮盘赌选中（`src/spd/aco_alns.py:1053`、`src/spd/aco_alns.py:1060`）
- 对应代码位置：`src/spd/aco_alns.py:155` 到 `src/spd/aco_alns.py:171`

#### D2: 最差贡献移除 `d2_worst_removal`

- 操作定义：逐个尝试移除客户并计算目标改善值 `baseline - candidate_obj`，优先移除“移除后改善最大”的客户（`src/spd/aco_alns.py:174`）
- 触发条件：被 destroy 轮盘赌选中
- 对应代码位置：`src/spd/aco_alns.py:174` 到 `src/spd/aco_alns.py:208`

#### D3: 相关性移除 `d3_related_removal`

- 操作定义：随机选种子客户，按几何距离排序，移除与种子最相关（最近）的一组客户（`src/spd/aco_alns.py:211`）
- 触发条件：被 destroy 轮盘赌选中
- 对应代码位置：`src/spd/aco_alns.py:211` 到 `src/spd/aco_alns.py:243`

#### D4: 低在家概率移除 `d4_low_home_probability_removal`

- 操作定义：先按当前计划到达时隙计算客户在家概率，优先移除概率最低客户（`src/spd/aco_alns.py:246`）
- 触发条件：被 destroy 轮盘赌选中
- 对应代码位置：`src/spd/aco_alns.py:246` 到 `src/spd/aco_alns.py:289`

### 3.2 Repair 算子（R）

#### R1: 贪婪插入 `r1_greedy_insertion`

- 操作定义：对待插回客户枚举可行插入候选（卡车位插、插入已有 sortie、新建 sortie），选目标值最小方案（`src/spd/aco_alns.py:432`）
- 触发条件：被 repair 轮盘赌选中（`src/spd/aco_alns.py:1054`、`src/spd/aco_alns.py:1061`）
- 对应代码位置：`src/spd/aco_alns.py:432` 到 `src/spd/aco_alns.py:462`

#### R2: 后悔值插入 `r2_regret_insertion`

- 操作定义：对每个客户计算 `regret = second_best - best`，优先插入 regret 最大客户（`src/spd/aco_alns.py:465`）
- 触发条件：被 repair 轮盘赌选中
- 对应代码位置：`src/spd/aco_alns.py:465` 到 `src/spd/aco_alns.py:502`

#### R3: 时隙感知插入 `r3_timeslot_aware_insertion`

- 操作定义：候选解先最大化到达时隙在家概率 `p_i(t*)`，并以目标值最小做次级打破平局（`src/spd/aco_alns.py:505`）
- 触发条件：被 repair 轮盘赌选中
- 对应代码位置：`src/spd/aco_alns.py:505` 到 `src/spd/aco_alns.py:527`

### 3.3 Cross-group 算子（低频跨车辆对操作）

#### C1: `cross_pair_swap`

- 操作定义：在两个车辆对之间交换小规模客户子集，并重建两个 pair 的 truck_route/sorties（`src/spd/aco_alns.py:583`）
- 触发条件：`(iteration + 1) % cross_group_frequency == 0` 且随机选中（`src/spd/aco_alns.py:1103` 到 `src/spd/aco_alns.py:1109`）
- 接受条件：新解硬可行且 `new_obj < old_obj`（`src/spd/aco_alns.py:620` 到 `src/spd/aco_alns.py:628`）
- 对应代码位置：`src/spd/aco_alns.py:583` 到 `src/spd/aco_alns.py:629`

#### C2: `cross_pair_transfer`

- 操作定义：将一个客户从 pair A 转移到 pair B，再重建两个 pair（`src/spd/aco_alns.py:631`）
- 触发条件：同 C1（周期触发后随机选算子）
- 接受条件：新解硬可行且严格改进
- 对应代码位置：`src/spd/aco_alns.py:631` 到 `src/spd/aco_alns.py:671`

### 3.4 可行性修复机制（非经典 destroy/repair，但在当前实现中非常关键）

#### Fallback 重建 `_fallback_rebuild`

- 操作定义：若 repair 后出现未服务客户或 sortie 结构破坏，则对受影响 pair 做重建（基于 ETPRC 的 `_build_truck_route + _extract_sorties`）
- 触发条件：`has_unserved or has_damaged`（`src/spd/aco_alns.py:1083` 到 `src/spd/aco_alns.py:1087`）
- 对应代码位置：`src/spd/aco_alns.py:731` 到 `src/spd/aco_alns.py:842`

## 4. 解的评价方式（目标函数计算逻辑）

核心实现：`src/spd/core.py:729` (`compute_objective`)

计算流程：

- 遍历 `solution.used_vehicle_pairs` 计算固定成本、卡车边成本、sortie 能耗成本
- 计算每个客户到达晚于 `latest` 的迟到量并累计 `z_tw_penalty`
- 计算期望失败成本 `z_fail_expected`（由 `1-p_i(t*)` 与 `Z_fail(i)` 构成）
- 对未服务客户加重罚（2倍失败罚）
- 汇总为总目标值返回

辅助分解函数：`compute_objective_breakdown`（`src/spd/core.py:783`），输出可直接用于论文表格：

- `z_fixed`
- `z_truck`
- `z_drone`
- `z_fail_expected`
- `z_tw_penalty`
- `z_unserved_penalty`
- `z_operational`
- `z_total`

## 5. 选择 / 接受准则

### 5.1 选择准则

- ALNS 算子选择：destroy 与 repair 均为轮盘赌（`roulette_select`，`src/spd/aco_alns.py:1038`）
- ACO 节点选择：蚂蚁构造下一客户也用轮盘赌（`src/spd/aco_alns.py:1799`）
- Cross-group 算子选择：周期触发后随机选择（`random_state.choice`，`src/spd/aco_alns.py:1108`）

### 5.2 接受准则（ALNS）

- 若 `delta < 0`，直接接受（`src/spd/aco_alns.py:1190` 到 `src/spd/aco_alns.py:1196`）
- 否则使用 Metropolis：
  `P_accept = exp(-delta / T)`（`src/spd/aco_alns.py:1197`）
- 抽样通过则接受更差解（`src/spd/aco_alns.py:1198` 到 `src/spd/aco_alns.py:1200`）

### 5.3 自适应权重更新

- reward 三档：`reward_global_best` / `reward_improve` / `reward_accept_worse`
- 权重更新公式：
  `w <- (1-lambda) * w + lambda * reward`
  （`src/spd/aco_alns.py:1255` 到 `src/spd/aco_alns.py:1262`）
- 温度退火：`T <- T * sa_cooling_rate`（`src/spd/aco_alns.py:1264`）

### 5.4 信息素学习（ACO）

- 蒸发：`tau <- tau * (1-rho)`（`src/spd/aco_alns.py:2016` 到 `src/spd/aco_alns.py:2018`）
- 沉积：`delta_tau = scale * Q / objective`（`src/spd/aco_alns.py:2038`）
- 同时对“本轮最优”和“全局最优”进行沉积（`src/spd/aco_alns.py:2046` 到 `src/spd/aco_alns.py:2047`）
- 限幅：`tau` 被裁剪到 `[tau_min, tau_max]`

## 6. 终止条件

### 6.1 ALNS 终止

- 固定迭代次数：`for iteration in range(params.alns.iterations)`（`src/spd/aco_alns.py:1052`）

### 6.2 ACO+ALNS 终止

- ACO 外层最多 `max_iterations`（`src/spd/aco_alns.py:1640`、`src/spd/aco_alns.py:1673`）
- 若连续无改进计数达到 `no_improve_max`，提前终止（`src/spd/aco_alns.py:1767`）

### 6.3 PureACO 终止

- 固定 `max_iterations`（`src/spd/aco_alns.py:2177`、`src/spd/aco_alns.py:2216`）
- 无改进时不提前退出，而是触发“部分信息素重启”后继续（`src/spd/aco_alns.py:2329` 到 `src/spd/aco_alns.py:2340`）

## 7. 完整伪代码

```text
Algorithm A: Offline Solve (ACO + ALNS)
Input: instance, params, all-home status, RNG
Output: best feasible offline solution

1  S0 <- ETPRC.build_initial_solution(...)
2  z0 <- objective(S0, beta=beta_init)
3  Initialize pheromone tau, state.best <- (S0, z0), beta <- beta_init
4  no_improve <- 0
5  for aco_iter = 1 .. max_iterations:
6      Build ant argument list (tau, beta, seed, global_best, ...)
7      for each ant (parallel or serial):
8          Sa <- construct_ant_solution(...)          # ACO constructive phase
9          S'a <- ALNS.optimize(Sa, global_best=z_best, beta=state.beta)
10         z'a <- objective(S'a, beta=state.beta)
11     Pick iteration-best (S_iter, z_iter)
12     if z_iter < z_best:
13         update global best; no_improve <- 0
14     else:
15         no_improve <- no_improve + 1
16     update_pheromone(tau, iteration_best, global_best)
17     beta <- min(beta * gamma_beta, beta_max)
18     if no_improve >= no_improve_max: break
19 return global_best_solution

Algorithm B: ALNS.optimize
Input: initial_solution, params, beta(current ACO), global_best(optional)
Output: best local feasible solution

1  S_cur <- deepcopy(initial_solution); z_cur <- objective(S_cur, beta)
2  S_best <- S_cur; z_best <- z_cur
3  Initialize destroy/repair weights; T <- sa_initial_temperature
4  for iter = 1 .. alns.iterations:
5      d <- roulette_select(destroy_weights)
6      r <- roulette_select(repair_weights)
7      q <- randint(remove_count_min, remove_count_max)
8      P <- Destroy_d(S_cur, q)
9      S_new <- Repair_r(P)
10     if (unserved exists OR damaged sortie): fallback_rebuild(S_new)
11     if periodic cross trigger: apply random cross-group operator
12     if not structural_and_physical_check(S_new): reject as invalid; reward=0
13     else:
14         z_new <- objective(S_new, beta)
15         delta <- z_new - z_cur
16         if delta < 0: accept
17         else accept with probability exp(-delta / T)
18         assign reward by {global_best, improve, accept_worse}
19     if accepted: S_cur <- S_new; z_cur <- z_new
20     if accepted and hard-feasible and z_cur < z_best: update S_best
21     update selected operator weights by reaction_factor and reward
22     T <- T * sa_cooling_rate
23 return S_best

Algorithm C: ACO construct_ant_solution
Input: groups from customer grouping, pheromone tau, heuristic weights
Output: complete candidate solution (or ETPRC fallback solution)

1  for each customer group:
2      route <- [depot], remaining <- group
3      while remaining not empty:
4          build candidate list by nearest distance
5          compute transition weight:
6              weight = (tau_ij ^ alpha_aco) * (eta_ij ^ beta_aco)
7          eta_ij combines distance slack, time-window slack, capacity slack
8          roulette select next customer
9          append to route and update time/load state
10     append depot, then call ETPRC._extract_sorties(route)
11 merge all pair solutions
12 if any unserved customer: return ETPRC.build_initial_solution(...)
13 return candidate solution
```

## 8. 关键参数列表（离线相关）

### 8.1 ACO 参数（`src/spd/config.py:89`）

- `max_iterations`：ACO 外层迭代上限
- `ant_count`：每轮蚂蚁数
- `alpha_aco`：信息素指数
- `beta_aco`：启发式信息指数
- `evaporation_rate`：蒸发率
- `pheromone_q`：沉积强度
- `pheromone_init`：初始信息素
- `no_improve_max`：无改进提前停止阈值
- `candidate_list_size`：候选邻域大小
- `tau_min` / `tau_max`：信息素边界
- `stagnation_restart_threshold` / `stagnation_restart_ratio`：停滞重启参数
- `global_best_weight`：全局最优沉积权重
- `eta_weight_distance` / `eta_weight_time_window` / `eta_weight_capacity`：启发式 eta 加权

### 8.2 ALNS 参数（`src/spd/config.py:112`）

- `iterations`：ALNS 内循环迭代数
- `remove_count_min` / `remove_count_max`：每次 destroy 移除规模范围
- `sa_initial_temperature` / `sa_cooling_rate`：退火初温与冷却率
- `operator_weight_init`：初始算子权重
- `reward_global_best` / `reward_improve` / `reward_accept_worse`：奖励档位
- `reaction_factor`：权重更新反应系数
- `cross_group_frequency`：跨 pair 算子触发周期
- `cross_group_remove_count`：跨 pair 操作规模

### 8.3 beta 调度参数（`src/spd/config.py:72`）

- `beta_init`：初始时间窗惩罚系数
- `beta_max`：上界
- `gamma_beta`：外层迭代乘性增长系数

### 8.4 ETPRC 相关参数（离线初解，`src/spd/config.py:130`）

- `k_cluster`、`max_iter_kmeans`
- `omega_distance`、`omega_time_window`、`omega_balance`

### 8.5 当前样例配置参考（`examples/configs/config_small.json`）

- `aco.max_iterations=20`
- `aco.ant_count=5`
- `alns.iterations=50`
- `alns.sa_initial_temperature=100.0`
- `alns.sa_cooling_rate=0.98`
- `beta_schedule: beta_init=10.0, beta_max=1000.0, gamma_beta=1.1`

## 9. 论文写作结构建议（新的离线算法章节）

基于当前代码，建议把论文中“离线阶段算法”改为下面结构：

- 3.1 离线问题定义与目标函数
- 3.2 ETPRC 初始解构造（分组、初始卡车路径、sortie 提取）
- 3.3 ACO 外层构造与信息素学习机制
- 3.4 ALNS 局部改进机制
- 3.5 邻域算子设计（D1-D4, R1-R3, Cross-group）
- 3.6 接受准则与自适应权重更新（SA + reward + reaction_factor）
- 3.7 可行性保障机制（fallback 重建 + 结构/物理约束检查）
- 3.8 终止条件、复杂度与并行实现说明
- 3.9 消融实验设置（`ACO+ALNS` vs `PureACO`）

与你旧框架的映射关系（基于当前代码）：

- `Destroy operators`：保留且更明确（D1-D4）
- `Repair operators`：保留且扩展为三种插入逻辑（R1-R3）
- `Pheromone-Based Learning`：保留，在 ACO 外层 `update_pheromone`
- `Feasibility Repair Operators`：旧式独立模块已被“fallback 重建 + 约束检查”替代
- `Adaptive weight adjustment`：保留，在 ALNS 主循环中更新
- `Local search`：保留，即 ALNS 内循环
- 新增重点：`cross_pair_swap/transfer`、动态 `beta` 调度、并行蚂蚁、MMAS 风格 tau 边界与停滞重启

## 10. 不确定点与代码层面观察

- `pure_alns` 分支在 `src/spd/main.py:195` 尝试动态导入 `PureALNSOptimizer`，但当前代码库未找到该类定义；此分支可能不可直接运行。
- `tests/test_pure_aco.py:99` 引用了 `examples/configs/config_pure_aco_small.json`，但仓库当前未发现该文件。
- 多处中文注释在当前文件编码下显示乱码；本文以可执行逻辑为准进行解释，不依赖乱码注释文本。
- `README.md` 仍描述为 skeleton/contract 状态，与当前已实现代码不完全一致；论文应以 `src/spd/*.py` 实现为唯一事实来源。
"""


def parse_markdown(md_text: str) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    lines = md_text.splitlines()
    in_code = False
    code_lines: list[str] = []
    para_lines: list[str] = []
    list_items: list[str] = []

    def flush_paragraph() -> None:
        nonlocal para_lines
        if para_lines:
            text = " ".join(part.strip() for part in para_lines if part.strip())
            if text:
                blocks.append({"type": "paragraph", "text": text})
        para_lines = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            blocks.append({"type": "list", "items": list_items[:]})
        list_items = []

    for raw in lines + [""]:
        line = raw.rstrip("\n")
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            flush_list()
            if not in_code:
                in_code = True
                code_lines = []
            else:
                blocks.append({"type": "code", "text": "\n".join(code_lines)})
                in_code = False
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            flush_paragraph()
            flush_list()
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            flush_list()
            blocks.append({"type": "heading", "level": 1, "text": stripped[2:].strip()})
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            flush_list()
            blocks.append({"type": "heading", "level": 2, "text": stripped[3:].strip()})
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            flush_list()
            blocks.append({"type": "heading", "level": 3, "text": stripped[4:].strip()})
            continue
        if stripped.startswith("#### "):
            flush_paragraph()
            flush_list()
            blocks.append({"type": "heading", "level": 4, "text": stripped[5:].strip()})
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            list_items.append(stripped[2:].strip())
            continue

        para_lines.append(line)

    return blocks


def render_inline_html(text: str) -> str:
    parts = text.split("`")
    rendered: list[str] = []
    for idx, part in enumerate(parts):
        if idx % 2 == 0:
            rendered.append(escape(part))
        else:
            rendered.append(f"<code>{escape(part)}</code>")
    return "".join(rendered)


def render_html(blocks: list[dict[str, object]]) -> str:
    body_lines: list[str] = []
    for block in blocks:
        btype = block["type"]
        if btype == "heading":
            level = int(block["level"])
            level = min(max(level, 1), 4)
            body_lines.append(f"<h{level}>{render_inline_html(str(block['text']))}</h{level}>")
        elif btype == "paragraph":
            body_lines.append(f"<p>{render_inline_html(str(block['text']))}</p>")
        elif btype == "list":
            items = block["items"]
            body_lines.append("<ul>")
            for item in items:  # type: ignore[assignment]
                body_lines.append(f"  <li>{render_inline_html(str(item))}</li>")
            body_lines.append("</ul>")
        elif btype == "code":
            body_lines.append(f"<pre><code>{escape(str(block['text']))}</code></pre>")

    content = "\n".join(body_lines)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Step2 Offline Optimization v3</title>
<style>
body {{
  margin: 0;
  background: #f4f6f8;
  color: #1f2933;
  font-family: "Times New Roman", Georgia, "Noto Serif SC", serif;
  line-height: 1.75;
}}
main {{
  max-width: 1160px;
  margin: 30px auto;
  background: #ffffff;
  border: 1px solid #d9e2ec;
  box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
  padding: 36px 46px;
}}
h1, h2, h3, h4 {{
  color: #102a43;
  line-height: 1.35;
  margin-top: 1.4rem;
}}
h1 {{
  font-size: 2rem;
  border-bottom: 3px solid #d9e2ec;
  padding-bottom: 0.5rem;
  margin-top: 0.2rem;
}}
h2 {{ font-size: 1.55rem; margin-top: 2rem; }}
h3 {{ font-size: 1.25rem; }}
h4 {{ font-size: 1.08rem; }}
p {{
  margin: 0.5rem 0 0.9rem 0;
  text-align: justify;
}}
ul {{
  margin: 0.35rem 0 0.95rem 0;
  padding-left: 1.4rem;
}}
li {{ margin: 0.22rem 0; }}
code {{
  background: #eef2f7;
  border-radius: 4px;
  padding: 1px 5px;
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.93em;
}}
pre {{
  white-space: pre-wrap;
  background: #f7fafc;
  border: 1px solid #cbd5e0;
  border-left: 4px solid #486581;
  padding: 12px 14px;
  margin: 0.9rem 0 1.05rem 0;
  font-family: Consolas, "Courier New", monospace;
  font-size: 0.92rem;
  line-height: 1.55;
}}
</style>
</head>
<body>
<main>
{content}
</main>
</body>
</html>
"""


def strip_inline_markers(text: str) -> str:
    return text.replace("`", "")


def make_docx_paragraph(text: str, style: str = "Normal", preserve: bool = False) -> str:
    if text == "":
        return "<w:p/>"
    style_xml = f"<w:pPr><w:pStyle w:val=\"{escape(style)}\"/></w:pPr>"
    space_attr = " xml:space=\"preserve\"" if preserve else ""
    return (
        "<w:p>"
        f"{style_xml}"
        "<w:r>"
        f"<w:t{space_attr}>{escape(text)}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def build_docx(blocks: list[dict[str, object]], output_path: Path) -> None:
    body_paras: list[str] = []

    for block in blocks:
        btype = block["type"]
        if btype == "heading":
            level = int(block["level"])
            level = min(max(level, 1), 3)
            style = f"Heading{level}"
            body_paras.append(make_docx_paragraph(strip_inline_markers(str(block["text"])), style=style))
        elif btype == "paragraph":
            body_paras.append(make_docx_paragraph(strip_inline_markers(str(block["text"])), style="Normal"))
        elif btype == "list":
            for item in block["items"]:  # type: ignore[index]
                body_paras.append(make_docx_paragraph("• " + strip_inline_markers(str(item)), style="Normal"))
        elif btype == "code":
            code_text = str(block["text"])
            if code_text == "":
                body_paras.append(make_docx_paragraph("", style="Code"))
            else:
                for code_line in code_text.splitlines():
                    body_paras.append(make_docx_paragraph(code_line, style="Code", preserve=True))
        body_paras.append(make_docx_paragraph(""))

    body_paras.append(
        "<w:sectPr>"
        "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )

    document_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
        "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
        "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
        "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
        "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
        "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
        "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
        "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
        "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
        "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
        "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
        "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
        "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
        "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
        "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" "
        "mc:Ignorable=\"w14 wp14\">"
        "<w:body>"
        + "".join(body_paras)
        + "</w:body></w:document>"
    )

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="30"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:next w:val="Normal"/>
    <w:qFormat/>
    <w:rPr><w:b/><w:sz w:val="26"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Code">
    <w:name w:val="Code"/>
    <w:basedOn w:val="Normal"/>
    <w:rPr>
      <w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>
      <w:sz w:val="20"/>
    </w:rPr>
  </w:style>
</w:styles>
"""

    content_types_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

    document_rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>
"""

    app_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
</Properties>
"""

    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    core_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>step2_offline_optimizationv3</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
"""

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("docProps/app.xml", app_xml)
        zf.writestr("docProps/core.xml", core_xml)
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", styles_xml)
        zf.writestr("word/_rels/document.xml.rels", document_rels_xml)


def main() -> None:
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    docs_dir = project_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    html_path = docs_dir / "step2_offline_optimizationv3.html"
    docx_path = docs_dir / "step2_offline_optimizationv3.docx"

    blocks = parse_markdown(MARKDOWN)
    html_text = render_html(blocks)
    html_path.write_text(html_text, encoding="utf-8")
    build_docx(blocks, docx_path)

    print(f"[OK] HTML generated: {html_path}")
    print(f"[OK] DOCX generated: {docx_path}")


if __name__ == "__main__":
    main()
