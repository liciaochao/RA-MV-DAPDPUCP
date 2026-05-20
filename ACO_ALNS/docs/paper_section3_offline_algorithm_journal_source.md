# 3. Offline Optimization Algorithm

This section presents the offline optimization procedure in a journal-style narrative consistent with the current implementation. Section 3.2 (ETPRC initial construction) is intentionally omitted from this chapter because it is documented separately; here, ETPRC is treated as a provided initializer and as a rebuild primitive.

## 3.1 Offline Objective for Algorithmic Optimization

The full optimization model has been formally defined in the earlier problem formulation section. In the implemented offline algorithm, the optimization target is operationalized as:

```text
min Z = Z_fixed + Z_truck + Z_drone + Z_fail_expected + beta * Z_tw_penalty + Z_unserved_penalty
```

where:

- `Z_fixed`: activated vehicle-pair fixed cost.
- `Z_truck`: truck travel cost over used arcs.
- `Z_drone`: drone energy cost aggregated over all sorties.
- `Z_fail_expected`: expected failed-service cost weighted by arrival-slot home probability.
- `Z_tw_penalty`: soft tardiness penalty for time-window violations.
- `Z_unserved_penalty`: explicit completeness penalty to suppress false improvements via customer dropping.

The time-window penalty coefficient is dynamically scheduled over ACO outer iterations:

```text
beta <- min(beta * gamma_beta, beta_max)
```

This schedule provides early-stage exploration and late-stage exploitation with stronger temporal discipline.

Recommended figure placement:
- Figure 3-1: Objective decomposition and information flow from route/sortie/timing states to each cost term.

## 3.3 ACO Outer Construction and Pheromone Learning

The global search layer follows an ACO outer loop. At each outer iteration, multiple ants build candidate solutions; each candidate is refined by ALNS; the best outcome updates the pheromone matrix and global incumbent.

### 3.3.1 Ant Construction Policy

For each customer group, route extension follows a pheromone-heuristic transition rule:

```text
w_ij = tau_ij^alpha_aco * eta_ij^beta_aco
```

The heuristic term is a weighted composition of:

- distance slack,
- time-window slack,
- capacity slack.

Candidate selection is roulette-wheel based. If no feasible option exists in the near-candidate list, the candidate pool is expanded. If construction quality degrades (e.g., incomplete assignment), the implementation falls back to a robust ETPRC-based reconstruction path.

### 3.3.2 Pheromone Update Rule

After each outer iteration:

1. global evaporation is applied to all pheromone edges;
2. iteration-best solution contributes a deposit term;
3. global-best solution contributes an additional weighted deposit term;
4. pheromone values are clipped to `[tau_min, tau_max]`.

Conceptually:

```text
tau <- clip((1-rho)*tau + Delta_tau_iter + omega_g*Delta_tau_global, tau_min, tau_max)
```

Recommended figure placement:
- Figure 3-2: ACO iteration workflow (construction -> ALNS refinement -> selection -> pheromone update -> beta update).
- Figure 3-3: Example pheromone heatmap evolution between two consecutive outer iterations.

## 3.4 ALNS Local Improvement Mechanism

Each ant solution enters an ALNS inner loop for local refinement. In each ALNS iteration:

1. one destroy and one repair operator are selected by roulette-wheel sampling from adaptive weights;
2. a random removal size is sampled from `[remove_count_min, remove_count_max]`;
3. destroy and repair are executed;
4. fallback rebuild is triggered if structural damage or unserved customers appears;
5. periodic cross-group perturbation is optionally executed;
6. candidate acceptance is decided by simulated annealing;
7. selected operator weights and temperature are updated.

The implementation distinguishes between:

- current accepted state (which may be exploratory),
- best local return state (which must satisfy hard feasibility constraints).

Recommended figure placement:
- Figure 3-4: ALNS state-transition graph with acceptance/rejection branches.

## 3.5 Neighborhood Operator Design (D1-D4, R1-R3, Cross-group)

### 3.5.1 Destroy Operator Set

1. `D1 Random Removal`  
Randomly removes `q` customers.

2. `D2 Worst Removal`  
Ranks customers by marginal objective relief after hypothetical single-customer removal; removes highest-contribution customers.

3. `D3 Related Removal`  
Starts from a random seed and removes geographically related customers.

4. `D4 Low-Probability Removal`  
Computes planned arrival-slot home probability and removes the lowest-probability customers.

### 3.5.2 Repair Operator Set

1. `R1 Greedy Insertion`  
Evaluates feasible insertion candidates and picks the minimum-objective insertion at each step.

2. `R2 Regret Insertion`  
Uses regret priority:

```text
regret(i) = Z_second_best(i) - Z_best(i)
```

3. `R3 Timeslot-aware Insertion`  
Prioritizes larger arrival-slot home probability and breaks ties by objective value.

### 3.5.3 Cross-group Operators

1. `Cross-pair Swap`  
Swaps small customer subsets between two pairs and accepts only strict improvement under feasibility checks.

2. `Cross-pair Transfer`  
Transfers customers from one pair to another, rebuilds both pairs, and accepts only strict improvement under feasibility checks.

Recommended figure placement:
- Figure 3-5: Before/after neighborhood transformations for D/R/Cross operators.
- Figure 3-6: A two-pair swap/transfer case study visualization.

## 3.6 Acceptance Criterion and Adaptive Weight Update

### 3.6.1 Simulated Annealing Acceptance

Let:

```text
Delta = Z_new - Z_cur
```

Acceptance rule:

- if `Delta < 0`, accept deterministically;
- else accept with probability:

```text
P_accept = exp(-Delta / T)
```

Temperature cooling:

```text
T <- max(T * sa_cooling_rate, epsilon)
```

### 3.6.2 Reward Scheme

The selected operator pair receives:

- `sigma_1` for new global best,
- `sigma_2` for local improvement,
- `sigma_3` for accepted worsening move,
- `0` otherwise (rejected/invalid).

### 3.6.3 Adaptive Weight Update

For selected destroy/repair operators:

```text
w <- (1 - reaction_factor) * w + reaction_factor * reward
```

This implements online credit assignment and adaptive search bias rebalancing.

Recommended table placement:
- Table 3-1: Reward class, acceptance condition, and weight-update impact.

## 3.7 Feasibility Safeguard Mechanism

In the new implementation, feasibility is no longer described as separate standalone repair operators (e.g., dedicated drone-load-only module). Instead, a layered safeguard pipeline is used:

1. structural damage detection for sortie anchors/order consistency;
2. fallback rebuild for affected pairs via ETPRC primitives;
3. structural/physical pre-check before candidate objective evaluation;
4. hard-feasible gating for best-local promotion.

Thus, feasibility assurance is centralized into a robust, reusable mechanism rather than fragmented operator-specific patches.

Recommended figure placement:
- Figure 3-7: Feasibility pipeline (Detect -> Rebuild -> Pre-check -> Best-feasible gate).
- Figure 3-8: Damaged sortie structure vs. repaired structure example.

## 3.8 Stopping Rules, Complexity, and Parallelization

### 3.8.1 Stopping Rules

The offline ACO+ALNS solver terminates when either condition is met:

1. maximum ACO outer iterations reached;
2. no-improvement counter reaches `no_improve_max`.

Within each ant, ALNS runs for a fixed number of local iterations.

### 3.8.2 Complexity Characterization

Let:

- `N`: customer count,
- `A`: ant count,
- `I`: ACO outer iterations,
- `L`: ALNS inner iterations.

A practical decomposition is:

```text
T_offline ~= I * A * (T_construct(N) + T_alns(L, N)) + I * T_pheromone
```

The dominant runtime factors are candidate generation, feasibility evaluation, objective recomputation, and fallback-trigger frequency.

### 3.8.3 Parallelization Strategy

The implementation applies ant-level multiprocessing:

- parallel unit: one ant task (`construct + ALNS refine`);
- worker count: up to `min(ant_count, CPU-1)`;
- synchronization barrier: end of each ACO iteration;
- fault handling: serial fallback if parallel execution fails.

Recommended figure placement:
- Figure 3-9: Parallel ant execution timeline with per-iteration synchronization.

## Algorithm 1. Overall Offline Optimization (ACO + ALNS)

```text
Input: instance, parameters, all-home policy, random seed
Output: best feasible offline solution S_best

1: S0 <- ETPRC_InitialSolution(instance, parameters)
2: Z0 <- Objective(S0, beta_init)
3: Initialize pheromone Tau, beta <- beta_init
4: S_best <- S0, Z_best <- Z0, noImprove <- 0

5: for iterACO = 1..I_aco_max do
6:     Build ant tasks with shared Tau and current beta
7:     Run ant tasks in parallel (or serial fallback)
8:     for each ant a result do
9:         S_a_ref, Z_a_ref <- refined ant solution and objective
10:    end for
11:    Select iteration-best (S_iter, Z_iter)
12:    if Z_iter < Z_best then
13:        S_best <- S_iter; Z_best <- Z_iter; noImprove <- 0
14:    else
15:        noImprove <- noImprove + 1
16:    end if
17:    Tau <- PheromoneUpdate(Tau, S_iter, Z_iter, S_best, Z_best)
18:    beta <- min(beta * gamma_beta, beta_max)
19:    if noImprove >= no_improve_max then break
20: end for
21: return S_best
```

## Algorithm 2. ACO Constructive Procedure

```text
Input: grouped customers, Tau, heuristic weights
Output: complete candidate solution S

1: for each pair-group G do
2:     route <- [depot], remaining <- G
3:     while remaining not empty do
4:         Build near-candidate list C
5:         Compute transition weight w_ij for j in C
6:         Roulette-select next customer j*
7:         Append j* and update time/load states
8:     end while
9:     Close route and extract sorties
10: end for
11: if unserved exists then
12:     fallback to robust ETPRC construction
13: end if
14: return S
```

## Algorithm 3. ALNS Local Search Procedure

```text
Input: S_cur, Z_global, beta
Output: locally best feasible solution S_best

1: Initialize destroy/repair weights and temperature T
2: S_best <- S_cur
3: for k = 1..N_alns do
4:     Select destroy D and repair R by roulette
5:     Sample q in [q_min, q_max]
6:     S_part <- D(S_cur, q)
7:     S_new <- R(S_part)
8:     if unserved or structural damage then
9:         S_new <- FallbackRebuild(S_new)
10:    end if
11:    if periodic cross condition holds then
12:        S_new <- CrossGroupPerturbation(S_new)
13:    end if
14:    if precheck fails then reject and reward <- 0
15:    else
16:        Z_new <- Objective(S_new, beta)
17:        accept <- SA_Accept(Z_cur, Z_new, T)
18:        reward <- RewardClass(accept, Z_new, Z_global)
19:    end if
20:    if accept then S_cur <- S_new
21:    if accept and hard_feasible(S_cur) and better(S_cur, S_best) then
22:        S_best <- S_cur
23:    end if
24:    Update weights with reward
25:    T <- T * cooling_rate
26: end for
27: return S_best
```

## Algorithm 4. SA Acceptance and Adaptive Weight Update

```text
Input: Z_cur, Z_new, Z_global, T, wD, wR
Output: accept flag and updated weights

1: Delta <- Z_new - Z_cur
2: if Delta < 0 then
3:     accept <- true
4:     reward <- sigma_1 if Z_new < Z_global else sigma_2
5: else
6:     p <- exp(-Delta / T)
7:     accept <- (rand() < p)
8:     reward <- sigma_3 if accept else 0
9: end if
10: wD <- (1-lambda_w) * wD + lambda_w * reward
11: wR <- (1-lambda_w) * wR + lambda_w * reward
12: return accept, wD, wR
```

## Algorithm 5. Feasibility Safeguard and Fallback Rebuild

```text
Input: candidate solution S
Output: repaired candidate S'

1: Detect damaged pairs and unserved customer set
2: if no damage and no unserved then return S
3: Assign unserved customers to suitable pairs
4: for each affected pair do
5:     Rebuild truck route and sorties via ETPRC primitives
6:     if rebuilt structure still invalid then
7:         clear sorties and move affected customers to truck route
8:     end if
9: end for
10: Recompute global unserved set from current assignments
11: return S'
```

## Readability and Visual Communication Notes

To retain the visual clarity of the previous version (which used several dedicated feasibility-repair figures), the new chapter can use:

1. one unified feasibility-safeguard pipeline figure;
2. one before/after fallback-rebuild route-sortie figure;
3. one constraints-coverage matrix (pre-check vs hard-feasibility check vs soft-penalty objective handling);
4. one operator-transformation panel (D/R/Cross exemplars).

These visual anchors align better with the current implementation than separate old-style load/energy repair modules.
