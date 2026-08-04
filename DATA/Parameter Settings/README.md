# Parameter Tuning Experiment Results

This directory contains the results of a representative one-factor-at-a-time
(OFAT) parameter sensitivity study for the ACO-ALNS algorithm. This document
explains how the experiment was designed, how to read the files, and which
conclusions can and cannot be drawn from the data.

## 1. Scope of the Study

The study consists of 19 deterministic offline optimization runs:

- one representative small instance: `small_15c_1v_seed3.json`;
- one representative medium instance: `medium_30c_2v_seed3.json`;
- one representative large instance: `large_150c_9v_seed3.json`;
- a fixed algorithmic random seed of 0;
- five ACO outer iterations and five ants per outer iteration;
- one changed parameter per non-baseline run.

These runs are descriptive fixed-seed sensitivity experiments. They are useful
for screening parameter values on the selected instances, but they are not
independent stochastic replications. The results therefore do not provide
variance estimates, confidence intervals, or evidence that a selected value is
optimal across other instances or random seeds.

Although every configuration contains `monte_carlo.trial_count = 100`, all runs
in this directory use `deterministic = true`. The deterministic reporting path
does not execute Monte Carlo simulation or online replanning. Consequently,
these results evaluate offline optimization only.

## 2. Quick Reading Guide

1. Open `parameter_tuning_summary.csv` for the experiment-level comparison.
2. Identify the baseline for each scale: `S_N100`, `M_BASE`, and `L_N400`.
3. Use `optimized_objective` as the primary solution-quality measure.
4. Use `relative_gap_to_baseline_percent` to compare a run with its scale
   baseline. A negative value is better than the baseline.
5. Use `total_seconds` to examine the quality-time tradeoff.
6. Open a run directory to inspect its exact configuration, cost components,
   solution, convergence history, logs, and figures.

Do not interpret the last row of `convergence_data.json` as the final global
best solution. See Section 8 for the correct interpretation.

## 3. Experiment Matrix

### 3.1 Instance-independent factors

The following factors were evaluated on the representative medium instance.
The middle level is contained in `M_BASE` and is shared as the baseline for all
five factor comparisons.

| Factor | Configuration field | Evaluated levels | Baseline |
|---|---|---:|---:|
| Evaporation rate, `rho` | `aco.evaporation_rate` | 0.10, 0.20, 0.30 | 0.20 |
| Capacity heuristic weight, `w_cap` | `aco.eta_weight_capacity` | 0.70, 0.85, 1.00 | 0.85 |
| Global-best deposition weight, `omega_g` | `aco.global_best_weight` | 0.45, 0.55, 0.65 | 0.55 |
| SA cooling rate, `alpha_sa` | `alns.sa_cooling_rate` | 0.95, 0.98, 0.995 | 0.98 |
| Historical persistence, `lambda` | `1 - alns.reaction_factor` | 0.40, 0.60, 0.80 | 0.60 |

The implementation stores the adaptive-weight reaction factor rather than the
persistence factor directly:

```text
reaction_factor = 1 - lambda
```

Thus, the baseline `lambda = 0.60` is stored as
`alns.reaction_factor = 0.40`.

### 3.2 Instance-size-dependent factor

The inner ALNS iteration budget was evaluated separately by problem size.

| Scale | Evaluated `N_alns` levels | Baseline run |
|---|---:|---|
| Small | 75, 100, 125 | `S_N100` |
| Medium | 150, 200, 250 | `M_BASE` |
| Large | 50, 100, 400 | `L_N400` |

The unusually wide large-instance range preserves the nominal value 400 while
also measuring lower-cost time-budget alternatives.

### 3.3 Fixed parameters that were not tuned here

The following values were used by all runs but were not varied in this study:

- `alpha_aco = 1.0`;
- `beta_aco = 2.0`;
- distance heuristic weight `w_dist = 1.0`;
- SA initial temperature `T0 = 100`;
- ALNS rewards `33`, `13`, and `9`;
- ACO outer iterations `5`;
- ACO ant count `5`;
- online ALNS iterations `20`;
- online removal count `2`.

Their presence in the configurations shows which values were used; it does not
show that these values were calibrated or proven optimal by this experiment.

### 3.4 Offline removal cardinality

The offline removal cardinality is a dynamic integer parameter, not an OFAT
factor. At each ALNS iteration it is sampled from a discrete uniform
distribution:

```text
q_min = max(3, floor(0.05 * |C|))
q_max = min(15, ceil(0.15 * |C|))
q ~ DiscreteUniform(q_min, q_max)
```

The realized bounds in these configurations are:

| Scale | Number of customers | Removal range |
|---|---:|---:|
| Small | 15 | 3 |
| Medium | 30 | 3 to 5 |
| Large | 150 | 7 to 15 |

## 4. Run Naming Convention

Run identifiers encode the scale and varied parameter:

- `S_`, `M_`, and `L_` indicate small, medium, and large instances;
- `N075`, `N100`, and similar suffixes indicate `N_alns`;
- `RHO010` and `RHO030` indicate `rho = 0.10` and `0.30`;
- `WCAP070` and `WCAP100` indicate `w_cap = 0.70` and `1.00`;
- `OMEGA045` and `OMEGA065` indicate `omega_g = 0.45` and `0.65`;
- `SA095` and `SA0995` indicate `alpha_sa = 0.95` and `0.995`;
- `LAMBDA040` and `LAMBDA080` indicate `lambda = 0.40` and `0.80`;
- `M_BASE` is the shared medium-instance baseline.

## 5. Directory Contents

Each run directory contains the following 15 files.

| File | Description |
|---|---|
| `config_used.json` | Exact parameter configuration used by the solver |
| `instance_used.json` | Exact instance copy used by the solver |
| `status.json` | Completion status, return code, elapsed time, and configuration hash |
| `stdout.log` | Standard solver output |
| `stderr.log` | Runtime warnings and diagnostic messages |
| `objective_summary.json` | Initial objective, optimized objective, and percentage improvement |
| `cost_breakdown.json` | Initial and optimized objective components |
| `solution_detail.json` | Complete final route and customer assignment data |
| `customer_service_report.json` | Per-customer service mode and timing information |
| `convergence_data.json` | Detailed ACO-ALNS iteration records |
| `timing.json` | Initial-solution, optimization, and total solver times |
| `timing_report.json` | Duplicate reporting copy of `timing.json` |
| `convergence_curve.png` | Cumulative global-best convergence figure |
| `cost_breakdown_table.png` | Initial-versus-optimized cost table |
| `solution_comparison.png` | Initial and optimized solution comparison |

The root file `parameter_tuning_summary.csv` combines the key results from all
19 runs.

## 6. Summary Table Columns

| Column | Meaning |
|---|---|
| `order` | Planned sequential execution order |
| `run_id` | Unique experiment identifier |
| `scale` | Small, medium, or large instance class |
| `instance` | Input instance filename |
| `varied_parameter` | Factor changed from the scale baseline |
| `parameter_value` | Display value of the varied factor |
| `baseline_run` | Baseline used for the relative comparison |
| `status` | Final run status |
| `initial_objective` | Objective value before ACO-ALNS optimization |
| `optimized_objective` | Best feasible objective found by the run |
| `improvement_percent` | Improvement from initial to optimized solution |
| `relative_gap_to_baseline_percent` | Difference from the scale baseline |
| `total_seconds` | Solver-reported initial plus optimization time |
| `result_directory` | Absolute path to the detailed run directory |

The reported percentages are calculated as follows:

```text
improvement_percent =
    100 * (initial_objective - optimized_objective) / initial_objective

relative_gap_to_baseline_percent =
    100 * (optimized_objective - baseline_objective) / baseline_objective
```

A lower objective is better. Therefore:

- a positive `improvement_percent` is desirable;
- a negative `relative_gap_to_baseline_percent` is better than the baseline;
- a zero relative gap means the same objective as the baseline.

`total_seconds` is the solver's internal timing. `status.json` contains a
slightly larger wrapper elapsed time that also includes process startup,
reporting, plotting, and file-handling overhead.

## 7. Main Experimental Results

### 7.1 Medium-instance factors

| Factor | Level | Optimized objective | Gap to baseline | Interpretation |
|---|---:|---:|---:|---|
| `rho` | 0.10 | 55.295198 | 0.00% | Same objective as baseline |
| `rho` | 0.20 | 55.295198 | 0.00% | Baseline |
| `rho` | 0.30 | 80.588777 | +45.74% | Worse on this instance |
| `w_cap` | 0.70 | 80.588777 | +45.74% | Worse on this instance |
| `w_cap` | 0.85 | 55.295198 | 0.00% | Baseline |
| `w_cap` | 1.00 | 53.185904 | -3.81% | Best tested medium result |
| `omega_g` | 0.45 | 55.295198 | 0.00% | Same objective as baseline |
| `omega_g` | 0.55 | 55.295198 | 0.00% | Baseline |
| `omega_g` | 0.65 | 55.295198 | 0.00% | Same objective as baseline |
| `alpha_sa` | 0.95 | 80.488239 | +45.56% | Worse on this instance |
| `alpha_sa` | 0.98 | 55.295198 | 0.00% | Baseline |
| `alpha_sa` | 0.995 | 55.295198 | 0.00% | Same objective as baseline |
| `lambda` | 0.40 | 80.613541 | +45.79% | Worse on this instance |
| `lambda` | 0.60 | 55.295198 | 0.00% | Best tested persistence level |
| `lambda` | 0.80 | 80.533459 | +45.64% | Worse on this instance |

The clearest medium-instance improvement is `w_cap = 1.00`. The data also
support retaining `lambda = 0.60`. The tested `omega_g` levels cannot be
distinguished by objective value. Similarly, `rho = 0.10` and `0.20` tie, and
`alpha_sa = 0.98` and `0.995` tie.

### 7.2 ALNS iteration budget

| Scale | `N_alns` | Optimized objective | Solver time |
|---|---:|---:|---:|
| Small | 75 | 34.791426 | 42.43 s |
| Small | 100 | 34.791426 | 58.10 s |
| Small | 125 | 34.791426 | 66.06 s |
| Medium | 150 | 80.588777 | 401.10 s |
| Medium | 200 | 55.295198 | 1,025.53 s |
| Medium | 250 | 55.295198 | 1,450.41 s |
| Large | 50 | 415.414360 | 7,055.80 s |
| Large | 100 | 413.434335 | 14,841.44 s |
| Large | 400 | 325.891635 | 68,630.21 s |

For the selected small instance, all three budgets reach the same objective and
75 iterations require the least time. For the medium instance, 200 and 250
iterations tie in objective, while 200 is faster; 150 iterations are
insufficient for the same solution quality. For the large instance, 400
iterations give the best tested objective but require approximately 19.06
hours, so the quality improvement has a substantial computational cost.

Exact objective ties do not indicate copied experiments. The solution-detail
files and convergence histories are distinct across all 19 runs.

## 8. How to Read Convergence Data

`convergence_data.json` contains records from 25 ALNS trajectories: five ACO
outer iterations multiplied by five ants. For a run with `N_alns = N`, the
expected record count is:

```text
25 * N + 1
```

The additional record is the common initial solution.

Important field semantics:

- `iteration` is the ALNS iteration number within one ant trajectory;
- `aco_iteration` identifies the ACO outer iteration;
- `ant_index` identifies the ant;
- `current_cost` is the current simulated-annealing state;
- `best_cost` is the best feasible incumbent within that trajectory;
- `operator_name` identifies the destroy-repair operator pair;
- `operator_scores` contains adaptive operator weights after the iteration.

The simulated-annealing state may temporarily accept an infeasible exploratory
solution, while `best_cost` is updated only by hard-constraint-feasible
solutions. It is therefore possible for `best_cost` to be greater than
`current_cost` in an individual record. This is expected under the implemented
logging semantics and does not imply that the final solution is infeasible.

Records from different ants are concatenated. The final row belongs to the last
ant and is not necessarily the global best. To reconstruct global convergence,
compute the cumulative minimum of feasible `best_cost` values over the complete
record sequence. The minimum `best_cost` over the full file equals the
`optimized_objective` reported for every run. The supplied
`convergence_curve.png` files already plot the cumulative global best correctly.

## 9. Data Integrity and Validation Notes

The result set has the following structural properties:

- all 19 runs completed with return code 0;
- every run contains all 15 expected files;
- all JSON files parse successfully;
- all PNG files are valid and nonempty;
- summary objectives, cost totals, timing totals, and status records agree;
- copied configurations match their recorded SHA-256 hashes;
- copied instances match the source instances;
- every final solution serves all customers;
- all final time-window and unserved-customer penalties are zero.

The complete study consumed approximately 28.075 solver hours, compared with
an initial estimate of 15.714 hours. The estimate should therefore not be used
as an accurate predictor for future computational planning.

Every `stderr.log` contains repeated Python `runpy` warnings generated during
parallel worker startup. No traceback, exception, fatal error, or failed status
was found. The large-instance runs additionally contain candidate-repair
warnings stating that no feasible insertion was found:

| Run | Candidate insertion warnings |
|---|---:|
| `L_N050` | 58 |
| `L_N100` | 519 |
| `L_N400` | 6,906 |

These warnings describe rejected search candidates, not final unserved
customers. All final large-instance solutions serve all 150 customers.

## 10. Interpretation Limits

Use the results to describe local sensitivity on the selected representative
instances. Do not use them alone to claim:

- a statistically significant parameter effect;
- robust performance across instance seeds;
- robustness across algorithmic random seeds;
- a globally optimal joint parameter configuration;
- the absence of interactions between parameters;
- validated online or Monte Carlo performance.

OFAT changes one parameter while holding all others at baseline. A parameter
that performs well individually may behave differently when combined with
another changed parameter. For example, the present data do not test a joint
configuration containing both `w_cap = 1.00` and `rho = 0.10`.

For confirmatory calibration, rerun the baseline and leading candidates on
multiple instance seeds and multiple algorithmic seeds using paired seeds,
then report distributions such as the median, interquartile range, confidence
interval, and runtime distribution.

## 11. Reproducibility Notes

Each run preserves its exact configuration and input instance, which supports
result-level auditing. However, the result directories do not record a source
code commit, Python version, installed dependency versions, hardware details,
or worker count. The project directory is also not associated with a Git commit
in the archived results. Objective comparisons within this sequential study
are internally controlled, but exact reproduction on another environment may
require additional provenance information.

When reporting these results, describe the study as a representative,
deterministic, fixed-seed OFAT sensitivity analysis rather than a replicated
stochastic calibration experiment.
