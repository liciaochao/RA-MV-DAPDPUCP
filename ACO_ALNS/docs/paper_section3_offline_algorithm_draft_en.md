# 3. Offline Optimization Algorithm

This section presents the offline optimization algorithm implemented in the current codebase.  
Section 3.2 (ETPRC initial solution construction) is intentionally omitted here because it is already documented as an independent chapter. In this chapter, ETPRC is treated as a given initializer and as a rebuild primitive used by the offline optimizer.

## 3.1 Offline Objective for Algorithmic Optimization

The complete mathematical problem model has been introduced earlier. The offline algorithm, however, optimizes an implementation-level objective that explicitly couples operational cost, expected failure risk, soft time-window penalties, and completeness penalties:

\[
\min Z =
Z_{\text{fixed}} + Z_{\text{truck}} + Z_{\text{drone}} +
Z_{\text{fail}}^{\text{exp}} + \beta \, Z_{\text{tw}} + Z_{\text{unserved}}
\]

where:

\[
Z_{\text{fixed}} = c_{\text{pair}} \cdot |\mathcal{P}_{\text{used}}|
\]

\[
Z_{\text{truck}} = \sum_{(i,j)\in \mathcal{A}_T} c_T \, d_T(i,j)
\]

\[
Z_{\text{drone}} = \sum_{s\in \mathcal{S}} c_E \, E_s
\]

\[
Z_{\text{fail}}^{\text{exp}} = \sum_{i\in \mathcal{N}_{\text{served}}}
\bigl(1-p_i(t_i^\*)\bigr)\,Z_{\text{fail}}(i),
\quad
Z_{\text{fail}}(i)=2\,d_T(0,i)\,c_T
\]

\[
Z_{\text{tw}} = \sum_{i\in \mathcal{N}_{\text{served}}} \max(0, a_i-l_i)
\]

\[
Z_{\text{unserved}} = \sum_{i\in \mathcal{N}_{\text{unserved}}} 2\,Z_{\text{fail}}(i)
\]

The dynamic coefficient \(\beta\) is increased along outer ACO iterations:

\[
\beta \leftarrow \min(\gamma_\beta \beta, \beta_{\max})
\]

This design makes the early search more exploratory and the late search more time-window sensitive.

Suggested figure:
- Figure 3-1: Cost composition diagram of \(Z\), with arrows from route/sortie/timing states to each cost term.

## 3.3 ACO Outer Construction and Pheromone Learning

The outer layer is an Ant Colony Optimization (ACO) process. At each outer iteration, multiple ants construct candidate solutions, and each candidate is refined by ALNS (Section 3.4). The best ant outcome updates the global best and pheromone matrix.

### 3.3.1 Ant Construction Rule

For each customer group (provided by ETPRC grouping), an ant constructs a truck sequence incrementally. The transition weight from current node \(i\) to candidate \(j\) is:

\[
w_{ij} = \tau_{ij}^{\alpha_{\text{aco}}}\,\eta_{ij}^{\beta_{\text{aco}}}
\]

\[
\eta_{ij}=w_d \,\eta^{\text{dist}}_{ij}+w_{tw}\,\eta^{\text{tw}}_{ij}+w_{cap}\,\eta^{\text{cap}}_{ij}
\]

where:
- \(\tau_{ij}\) is pheromone.
- \(\eta^{\text{dist}}_{ij}\) rewards short travel distance.
- \(\eta^{\text{tw}}_{ij}\) rewards larger time-window slack.
- \(\eta^{\text{cap}}_{ij}\) rewards larger capacity slack.

Roulette-wheel sampling is applied on candidate weights. If no candidate is feasible in the near list, the algorithm expands the search pool, and falls back to robust defaults if needed.

### 3.3.2 Pheromone Update

After evaluating all ants in one outer iteration:
- Global evaporation:
\[
\tau_{ij}\leftarrow (1-\rho)\tau_{ij}
\]
- Iteration-best deposition:
\[
\Delta \tau_{ij}^{\text{iter}} = \frac{Q_{\text{aco}}}{Z_{\text{iter-best}}}
\]
- Global-best deposition:
\[
\Delta \tau_{ij}^{\text{global}} = \omega_g \frac{Q_{\text{aco}}}{Z_{\text{global-best}}}
\]
- Bounded update:
\[
\tau_{ij}\leftarrow \text{clip}\bigl(\tau_{ij}+\Delta \tau_{ij}^{\text{iter}}+\Delta \tau_{ij}^{\text{global}},\tau_{\min},\tau_{\max}\bigr)
\]

Suggested figures:
- Figure 3-2: ACO outer iteration workflow (construct -> refine -> select -> update pheromone -> update beta).
- Figure 3-3: Example pheromone heatmap over truck arcs before and after one iteration.

## 3.4 ALNS Local Improvement Mechanism

Each ant-constructed solution is passed to ALNS for local refinement. ALNS iteratively applies destroy-repair operations with simulated annealing acceptance and adaptive operator weighting.

At ALNS iteration \(k\):
1. Select one destroy operator and one repair operator by roulette-wheel probabilities from current weights.
2. Randomly draw removal size \(q \in [q_{\min}, q_{\max}]\).
3. Destroy part of the current assignment and repair it.
4. Optionally trigger cross-group operator by frequency rule.
5. Evaluate candidate under current \(\beta\) and acceptance policy.
6. Update solution state, best-feasible state, operator weights, and temperature.

The ALNS local search does not assume every accepted state is fully hard-feasible. Instead, it uses a safeguarded strategy where the returned best local solution is always checked against hard constraints.

Suggested figure:
- Figure 3-4: ALNS state transition diagram (Current -> Destroy -> Repair -> Check -> Accept/Reject -> Weight/Temperature update).

## 3.5 Neighborhood Operator Design (D1-D4, R1-R3, Cross-group)

### 3.5.1 Destroy Operators

1. D1 Random Removal  
Randomly removes \(q\) customers from the current assignment.

2. D2 Worst Removal  
Estimates per-customer contribution by objective improvement after single removal, then removes the most "harmful" customers.

3. D3 Related Removal  
Chooses a seed customer and removes geographically related customers near the seed.

4. D4 Low Home-Probability Removal  
Computes planned arrival slot probability \(p_i(t_i^\*)\), then removes customers with the lowest probabilities.

### 3.5.2 Repair Operators

1. R1 Greedy Insertion  
Enumerates feasible insertions and picks minimum-objective insertion each step.

2. R2 Regret Insertion  
Uses regret-\(k\) style logic with
\[
\text{regret}(i) = Z_{2nd}(i)-Z_{best}(i)
\]
and inserts the customer with largest regret first.

3. R3 Timeslot-aware Insertion  
Prefers candidates with higher \(p_i(t_i^\*)\); objective value is used as tie-breaker.

### 3.5.3 Cross-group Operators

1. Cross-pair Swap  
Swaps small customer subsets between two vehicle pairs, rebuilds both pairs, and accepts only if objective strictly improves and constraints hold.

2. Cross-pair Transfer  
Transfers one customer from one pair to another, rebuilds, and accepts only under strict improvement and feasibility.

Suggested figures:
- Figure 3-5: Operator gallery with before/after route-sortie sketches for D/R/Cross operators.
- Figure 3-6: One concrete swap/transfer case visualization between two vehicle pairs.

## 3.6 Acceptance Criterion and Adaptive Weight Update

### 3.6.1 Simulated Annealing Acceptance

Let \(\Delta = Z_{\text{new}} - Z_{\text{cur}}\).

- If \(\Delta < 0\), accept.
- Else accept with Metropolis probability:
\[
P_{\text{acc}} = \exp\left(-\frac{\Delta}{T}\right)
\]

Temperature cooling:
\[
T \leftarrow \max(T \cdot \alpha_{\text{cool}}, \epsilon)
\]

### 3.6.2 Reward Scheme

If accepted:
- \(\sigma_1\): new global best.
- \(\sigma_2\): local improvement.
- \(\sigma_3\): accepted worse move.

If rejected or invalid candidate: reward \(=0\).

### 3.6.3 Adaptive Operator Weights

For selected destroy/repair operators:

\[
w \leftarrow (1-\lambda_w)w + \lambda_w \cdot \text{reward}
\]

This allows online credit assignment and re-balancing between exploitation and exploration.

Suggested table:
- Table 3-1: Reward and update rules for \(\sigma_1,\sigma_2,\sigma_3\), with acceptance conditions.

## 3.7 Feasibility Safeguard Mechanism

In the current implementation, feasibility assurance is no longer handled by standalone "repair operators" such as separate drone-load or drone-energy repair modules. Instead, feasibility is enforced by a layered safeguard pipeline:

1. Structural damage detection  
Detects broken sortie anchors/order (launch-recovery consistency, overlap issues).

2. Fallback rebuild  
When unserved customers or damaged pair structures appear, affected pairs are rebuilt using ETPRC primitives.

3. Structural and physical pre-check in ALNS loop  
During local search, structural and physical validity are checked before objective evaluation.

4. Hard-feasible best-solution gating  
A candidate can be accepted as current state for exploration, but promotion to the best-local solution requires full hard-feasibility checks.

This design replaces monolithic, operator-specific feasibility repair by a general, robust, and modular safeguard architecture.

Suggested figures:
- Figure 3-7: Feasibility safeguard pipeline (Detect -> Rebuild -> Pre-check -> Hard-feasible best gating).
- Figure 3-8: Example of damaged sortie structure and rebuilt valid structure.

## 3.8 Stopping Rules, Complexity, and Parallelization

### 3.8.1 Stopping Rules

ACO + ALNS offline solver stops when one condition is met:

1. Maximum ACO iterations reached.
2. Consecutive non-improving outer iterations reaches threshold \(N_{\text{no-improve}}\).

Inside each ant, ALNS runs for a fixed number of local iterations \(N_{\text{alns}}\).

### 3.8.2 Complexity Discussion

Let:
- \(N\): number of customers.
- \(A\): number of ants.
- \(I\): number of ACO outer iterations.
- \(L\): ALNS iterations per ant.

A practical upper-bound style decomposition is:

\[
\mathcal{T}_{\text{offline}}
\approx
I \cdot A \cdot
\bigl(
\mathcal{T}_{\text{construct}}(N) +
\mathcal{T}_{\text{ALNS}}(L,N)
\bigr)
 + I \cdot \mathcal{T}_{\text{pheromone}}
\]

Construction is dominated by repeated candidate scoring and route extension (roughly superlinear in \(N\)); ALNS cost is dominated by repeated candidate generation, feasibility checks, and objective recomputation. In practice, operator choices and fallback frequency strongly affect runtime.

### 3.8.3 Parallelization Strategy

Ant-level parallelism is implemented by multiprocessing with spawn semantics:

- Parallel unit: one ant task (construct + ALNS refine).
- Worker count: \(\min(A, \text{CPU}-1)\).
- Synchronization: at the end of each outer iteration for best-solution and pheromone updates.
- Fault tolerance: if parallel execution fails, the system falls back to serial execution.

Suggested figure:
- Figure 3-9: Parallel execution timeline (ant workers per ACO iteration with synchronization barrier).

## Algorithm 1. Overall Offline Optimization (ACO + ALNS)

```text
Input: instance, parameters, all-home policy, random seed
Output: best feasible offline solution S_best

1: S0 <- ETPRC_InitialSolution(instance, parameters)
2: Z0 <- Objective(S0, beta_init)
3: Initialize pheromone matrix Tau, beta <- beta_init
4: S_best <- S0, Z_best <- Z0, noImprove <- 0

5: for iterACO = 1 to I_aco_max do
6:     Create ant tasks with shared Tau and current beta
7:     Run ants in parallel (or serial fallback):
8:         for each ant a do
9:             S_a <- ACO_Construct(instance, Tau, beta)
10:            S_a_ref <- ALNS_Refine(S_a, Z_best, beta)
11:            Z_a_ref <- Objective(S_a_ref, beta)
12:        end for

13:    Select iteration-best solution (S_iter, Z_iter)
14:    if Z_iter < Z_best then
15:        S_best <- S_iter, Z_best <- Z_iter, noImprove <- 0
16:    else
17:        noImprove <- noImprove + 1
18:    end if

19:    Tau <- PheromoneUpdate(Tau, S_iter, Z_iter, S_best, Z_best)
20:    beta <- min(beta * gamma_beta, beta_max)
21:    if noImprove >= noImprove_max then break
22: end for

23: return S_best
```

## Algorithm 2. ACO Constructive Mechanism

```text
Input: grouped customers, pheromone Tau, heuristic parameters
Output: complete candidate solution S

1: for each vehicle-pair group G do
2:     route <- [depot], remaining <- G
3:     while remaining not empty do
4:         Build candidate list C (near-first strategy)
5:         For each j in C, compute transition weight:
6:             w_ij = Tau_ij^alpha * eta_ij^beta
7:         Roulette-select next customer j*
8:         Append j* to route and update time/load states
9:     end while
10:    Append depot and extract sorties on route
11: end for
12: if unserved customers exist then
13:    fallback to robust initial solution builder
14: end if
15: return S
```

## Algorithm 3. ALNS Local Search Mechanism

```text
Input: initial solution S_cur, global-best objective Z_global, beta
Output: local best feasible solution S_best

1: Initialize destroy/repair weights and temperature T
2: S_best <- S_cur
3: for k = 1 to N_alns do
4:     Select destroy operator D and repair operator R by roulette
5:     q <- random integer in [q_min, q_max]
6:     S_part <- D(S_cur, q)
7:     S_new <- R(S_part)
8:     if unserved customers or damaged structure then
9:         S_new <- FallbackRebuild(S_new)
10:    end if
11:    if periodic cross condition then
12:        S_new <- CrossGroupOperator(S_new)
13:    end if

14:    if not structural-physical precheck(S_new) then
15:        reward <- 0; reject
16:    else
17:        Z_new <- Objective(S_new, beta)
18:        Decide acceptance by SA rule
19:        Assign reward by (global-best / improve / accept-worse / reject)
20:    end if

21:    if accepted then S_cur <- S_new
22:    if accepted and hard-feasible(S_cur) and better than S_best then
23:        S_best <- S_cur
24:    end if
25:    Update selected operator weights
26:    T <- T * cooling_rate
27: end for
28: return S_best
```

## Algorithm 4. Acceptance and Weight-Adaptation Rule

```text
Input: Z_cur, Z_new, Z_global, T, selected weights wD, wR
Output: accept/reject, updated weights

1: Delta <- Z_new - Z_cur
2: if Delta < 0 then
3:     accept <- true
4:     if Z_new < Z_global then reward <- sigma1
5:     else reward <- sigma2
6: else
7:     p <- exp(-Delta / T)
8:     accept <- (rand() < p)
9:     if accept then reward <- sigma3
10:    else reward <- 0
11: end if

12: wD <- (1 - lambda_w) * wD + lambda_w * reward
13: wR <- (1 - lambda_w) * wR + lambda_w * reward
14: return accept, wD, wR
```

## Algorithm 5. Feasibility Safeguard with Fallback Rebuild

```text
Input: candidate solution S
Output: repaired solution S'

1: Detect damaged pairs (broken sortie anchors/order) and unserved set
2: if no damage and no unserved then return S
3: Assign unserved customers to nearest/reasonable pairs
4: for each affected pair do
5:     Rebuild truck route and sorties using ETPRC primitives
6:     if rebuilt pair still structurally invalid then
7:         clear sorties and move affected customers back to truck service
8:     end if
9: end for
10: Recompute unserved customers from global assignment
11: return S'
```

## Writing Note for Readability

If you want to preserve the visual clarity you had in the previous "Feasibility Repair Operators" subsection, use the new visual focus below:

1. Replace operator-specific repair figures with one unified safeguard pipeline figure.
2. Add two route-sortie snapshots: before fallback rebuild and after fallback rebuild.
3. Add one constraints coverage map table:
   - rows: constraints (energy, drone load, truck load, sequence, time windows, depot return)
   - columns: precheck stage, full hard-feasibility stage, objective soft-penalty stage
4. Add one operator-impact figure set for D/R/Cross transformations.

This gives a clearer correspondence to the new implementation than the old three isolated repair-operator illustrations.
