from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from model_types import ProblemData, SortieCandidate

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover - runtime guard
    gp = None
    GRB = None
    _GUROBI_IMPORT_ERROR = exc
else:
    _GUROBI_IMPORT_ERROR = None


def check_gurobi_available() -> tuple[bool, str]:
    if gp is None:
        return False, f'gurobipy import failed: {_GUROBI_IMPORT_ERROR}'
    try:
        if hasattr(gp, 'grbVersion'):
            version_tuple = gp.grbVersion()
        else:
            version_tuple = gp.gurobi.version()
        version = '.'.join(map(str, version_tuple))
    except Exception as exc:  # pragma: no cover - defensive
        return False, f'gurobi runtime check failed: {exc}'
    return True, version


def _status_to_text(status_code: int, sol_count: int) -> str:
    if status_code == GRB.OPTIMAL:
        return 'optimal'
    if status_code == GRB.INFEASIBLE:
        return 'infeasible'
    if status_code == GRB.TIME_LIMIT:
        return 'feasible' if sol_count > 0 else 'timeout'
    if status_code in (GRB.SUBOPTIMAL, GRB.INTERRUPTED):
        return 'feasible' if sol_count > 0 else 'timeout'
    return 'timeout' if sol_count == 0 else 'feasible'


def _extract_truck_route(
    data: ProblemData,
    x_values: dict[tuple[int, int], float],
) -> list[int]:
    successor: dict[int, int] = {}
    for (i, j), value in x_values.items():
        if value > 0.5:
            successor[i] = j

    route = [data.depot_start]
    current = data.depot_start
    guard = 0
    while current != data.depot_end and guard <= len(data.nodes) + 5:
        nxt = successor.get(current)
        if nxt is None:
            break
        route.append(nxt)
        current = nxt
        guard += 1

    if route and route[-1] == data.depot_end:
        route[-1] = 0
    elif route[-1] != 0:
        route.append(0)
    return route


def _format_float(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _initial_sortie_load(data: ProblemData, customers: tuple[int, ...]) -> float:
    """Initial drone load at launch: delivery parcels only."""
    load = 0.0
    for customer_id in customers:
        customer_type = str(data.customer_type.get(customer_id, 'delivery')).lower()
        if customer_type == 'delivery':
            load += data.weights.get(customer_id, 0.0)
    return load


def _final_sortie_load(data: ProblemData, customers: tuple[int, ...]) -> float:
    """Remaining drone load right before recovery."""
    load = _initial_sortie_load(data, customers)
    for customer_id in customers:
        customer_type = str(data.customer_type.get(customer_id, 'delivery')).lower()
        weight = data.weights.get(customer_id, 0.0)
        if customer_type == 'delivery':
            load -= weight
        elif customer_type == 'pickup':
            load += weight
    return load


def solve_deterministic_milp(
    data: ProblemData,
    candidates: list[SortieCandidate],
    time_limit: int = 3600,
    mip_gap: float = 0.005,
    threads: int = 6,
    output_flag: int = 1,
) -> dict[str, Any]:
    if gp is None:
        raise RuntimeError(f'gurobipy import failed: {_GUROBI_IMPORT_ERROR}')

    model = gp.Model('TruckDroneDeterministic')

    C = list(data.customers)
    C_d = set(data.drone_customers)
    N = list(data.nodes)
    end = data.depot_end

    arc_list = [
        (i, j)
        for i in N
        for j in N
        if i != j and i != end and j != data.depot_start
    ]

    out_arcs: dict[int, list[tuple[int, int]]] = {i: [] for i in N}
    in_arcs: dict[int, list[tuple[int, int]]] = {i: [] for i in N}
    for arc in arc_list:
        out_arcs[arc[0]].append(arc)
        in_arcs[arc[1]].append(arc)

    K = list(range(len(candidates)))

    customer_to_candidates: dict[int, list[int]] = {i: [] for i in C}
    launch_to_candidates: dict[int, list[int]] = {i: [] for i in C}
    recovery_to_candidates: dict[int, list[int]] = {i: [] for i in N}
    for k, cand in enumerate(candidates):
        for cid in cand.customers:
            customer_to_candidates[cid].append(k)
        if cand.launch in launch_to_candidates:
            launch_to_candidates[cand.launch].append(k)
        recovery_to_candidates[cand.recovery].append(k)

    x = model.addVars(arc_list, vtype=GRB.BINARY, name='x')
    v = model.addVars(C, vtype=GRB.BINARY, name='v')
    p = model.addVars(C, vtype=GRB.BINARY, name='pass')
    u = model.addVars(C, lb=0.0, ub=len(C), vtype=GRB.CONTINUOUS, name='u')

    time_ub = float(data.depot_latest)
    max_candidate_duration = max((cand.duration for cand in candidates), default=0.0)
    return_ub = float(data.depot_latest + data.max_wait_time + max_candidate_duration)

    T = model.addVars(N, lb=0.0, ub=time_ub, vtype=GRB.CONTINUOUS, name='T')
    S = model.addVars(C, lb=0.0, ub=time_ub, vtype=GRB.CONTINUOUS, name='S')

    z = model.addVars(K, vtype=GRB.BINARY, name='z')
    T_launch = model.addVars(K, lb=0.0, ub=time_ub, vtype=GRB.CONTINUOUS, name='T_launch')
    T_drone_ret = model.addVars(K, lb=0.0, ub=return_ub, vtype=GRB.CONTINUOUS, name='T_drone_ret')
    W = model.addVars(K, lb=0.0, ub=data.max_wait_time, vtype=GRB.CONTINUOUS, name='W')
    H = model.addVars(K, lb=0.0, ub=return_ub, vtype=GRB.CONTINUOUS, name='H')
    E = model.addVars(K, lb=0.0, ub=data.battery_capacity, vtype=GRB.CONTINUOUS, name='E')

    # (C1) truck flow by pass[i]
    for i in C:
        model.addConstr(gp.quicksum(x[a] for a in out_arcs[i]) == p[i], name=f'C1_out_{i}')
        model.addConstr(gp.quicksum(x[a] for a in in_arcs[i]) == p[i], name=f'C1_in_{i}')

    # (C2) depot start/end
    model.addConstr(gp.quicksum(x[a] for a in out_arcs[data.depot_start]) == 1, name='C2_start')
    model.addConstr(gp.quicksum(x[a] for a in in_arcs[end]) == 1, name='C2_end')

    # C4 and pass-link
    for i in C:
        if i in C_d:
            model.addConstr(
                v[i] + gp.quicksum(z[k] for k in customer_to_candidates[i]) == 1,
                name=f'C4_service_{i}',
            )
        else:
            model.addConstr(v[i] == 1, name=f'C4_truck_only_{i}')
        model.addConstr(p[i] >= v[i], name=f'C7_pass_from_v_{i}')

    for i in C:
        for k in launch_to_candidates[i]:
            model.addConstr(p[i] >= z[k], name=f'C7_pass_launch_{i}_{k}')
        for k in recovery_to_candidates[i]:
            model.addConstr(p[i] >= z[k], name=f'C7_pass_recovery_{i}_{k}')

    # (C3) MTZ subtour elimination
    n = len(C)
    for i in C:
        model.addConstr(u[i] >= p[i], name=f'C3_u_lb_{i}')
        model.addConstr(u[i] <= n * p[i], name=f'C3_u_ub_{i}')

    for i in C:
        for j in C:
            if i == j:
                continue
            if (i, j) not in x:
                continue
            model.addConstr(
                u[i] - u[j] + n * x[(i, j)] <= n - 1 + n * (2 - p[i] - p[j]),
                name=f'C3_mtz_{i}_{j}',
            )

    # (C9) launch before recovery when both are customer nodes.
    big_u = float(n + 2)
    for k, cand in enumerate(candidates):
        if cand.launch in C and cand.recovery in C:
            model.addConstr(
                u[cand.launch] + 1 <= u[cand.recovery] + big_u * (1 - z[k]),
                name=f'C9_{k}',
            )

    # (C10) disjoint sortie intervals (stronger than no-crossing; disallows overlap).
    pair_indices = [(k1, k2) for k1 in K for k2 in K if k1 < k2]
    order = model.addVars(pair_indices, vtype=GRB.BINARY, name='order')

    def start_expr(k: int):
        launch = candidates[k].launch
        return 0.0 if launch == data.depot_start else u[launch]

    def end_expr(k: int):
        recovery = candidates[k].recovery
        return float(n + 1) if recovery == end else u[recovery]

    for k1, k2 in pair_indices:
        model.addConstr(
            end_expr(k1)
            <= start_expr(k2)
            + big_u * (1 - order[(k1, k2)])
            + big_u * (2 - z[k1] - z[k2]),
            name=f'C10_a_{k1}_{k2}',
        )
        model.addConstr(
            end_expr(k2)
            <= start_expr(k1)
            + big_u * order[(k1, k2)]
            + big_u * (2 - z[k1] - z[k2]),
            name=f'C10_b_{k1}_{k2}',
        )

    # (C11/C12/C15) sortie energy/time/wait/hover
    big_t = max(10000.0, return_ub + data.depot_latest + 60.0)
    for k, cand in enumerate(candidates):
        hover_power_per_min = data.eta_wh_per_kg_min * (
            data.drone_empty_weight + _final_sortie_load(data, cand.customers)
        )

        model.addConstr(
            E[k] == cand.energy_wh * z[k] + hover_power_per_min * H[k],
            name=f'C12_energy_{k}',
        )
        model.addConstr(E[k] <= data.battery_capacity * z[k], name=f'C12_battery_{k}')

        if cand.launch == data.depot_start:
            model.addConstr(
                T_launch[k] >= T[data.depot_start] - big_t * (1 - z[k]),
                name=f'C15_launch0_lb_{k}',
            )
            model.addConstr(
                T_launch[k] <= T[data.depot_start] + big_t * (1 - z[k]),
                name=f'C15_launch0_ub_{k}',
            )
        else:
            model.addConstr(
                T_launch[k] >= T[cand.launch] - big_t * (1 - z[k]),
                name=f'C15_launch_lb_{k}',
            )
            model.addConstr(
                T_launch[k] <= T[cand.launch] + big_t * (1 - z[k]),
                name=f'C15_launch_ub_{k}',
            )

        model.addConstr(
            T_drone_ret[k] >= T_launch[k] + cand.duration - big_t * (1 - z[k]),
            name=f'C15_ret_lb_{k}',
        )
        model.addConstr(
            T_drone_ret[k] <= T_launch[k] + cand.duration + big_t * (1 - z[k]),
            name=f'C15_ret_ub_{k}',
        )

        # Drone hover at recovery when truck arrives later than drone.
        model.addConstr(
            H[k] >= T[cand.recovery] - T_drone_ret[k] - big_t * (1 - z[k]),
            name=f'C15_hover_lb_{k}',
        )
        model.addConstr(H[k] <= return_ub * z[k], name=f'C15_hover_ub_{k}')

        model.addConstr(
            W[k] >= T_drone_ret[k] - T[cand.recovery] - big_t * (1 - z[k]),
            name=f'C15_wait_lb_{k}',
        )
        model.addConstr(W[k] <= data.max_wait_time * z[k], name=f'C15_wait_ub_{k}')

    # (C13) truck time propagation + waiting at recovery nodes.
    model.addConstr(T[data.depot_start] == data.depot_earliest, name='C14_start_time')

    for (i, j) in arc_list:
        wait_i = gp.quicksum(W[k] for k in recovery_to_candidates.get(i, []))
        travel_ij = data.distance[(i, j)] / data.truck_speed
        service_i = data.service_times.get(i, 0.0)
        model.addConstr(
            T[j]
            >= T[i] + service_i + wait_i + travel_ij - big_t * (1 - x[(i, j)]),
            name=f'C13_{i}_{j}',
        )

    model.addConstr(T[end] <= data.depot_latest, name='C14_end_time')

    # Service-time linking for customer time windows.
    for i in C:
        model.addConstr(S[i] >= T[i] - big_t * (1 - v[i]), name=f'C14_service_truck_lb_{i}')
        model.addConstr(S[i] <= T[i] + big_t * (1 - v[i]), name=f'C14_service_truck_ub_{i}')

        for k in customer_to_candidates[i]:
            offset = candidates[k].service_offsets[i]
            model.addConstr(
                S[i] >= T_launch[k] + offset - big_t * (1 - z[k]),
                name=f'C14_service_drone_lb_{i}_{k}',
            )
            model.addConstr(
                S[i] <= T_launch[k] + offset + big_t * (1 - z[k]),
                name=f'C14_service_drone_ub_{i}_{k}',
            )

        tw_lb, tw_ub = data.time_windows[i]
        model.addConstr(S[i] >= tw_lb, name=f'C14_tw_lb_{i}')
        model.addConstr(S[i] <= tw_ub, name=f'C14_tw_ub_{i}')

    # Conservative truck load cap.
    model.addConstr(
        gp.quicksum(data.weights[i] * v[i] for i in C) <= data.truck_capacity,
        name='C_load_total',
    )

    # Objective: fixed + truck + drone energy
    z_truck_expr = gp.quicksum(
        data.truck_cost_per_km * data.distance[(i, j)] * x[(i, j)]
        for (i, j) in arc_list
    )
    z_drone_expr = gp.quicksum(data.drone_cost_per_wh * E[k] for k in K)
    objective = data.fixed_pair_cost + z_truck_expr + z_drone_expr
    model.setObjective(objective, GRB.MINIMIZE)

    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', mip_gap)
    model.setParam('Threads', threads)
    model.setParam('OutputFlag', output_flag)

    model.optimize()

    status_text = _status_to_text(model.Status, model.SolCount)

    objective_value: float | None = None
    gap_percent: float | None = None
    z_truck_value: float | None = None
    z_drone_value: float | None = None
    x_values: dict[tuple[int, int], float] = {arc: 0.0 for arc in arc_list}

    truck_customers: list[int] = []
    sorties_output: list[dict[str, Any]] = []
    route_output: list[int] = [0, 0]
    served_customers: list[int] = []
    unserved_customers: list[int] = list(C)
    diagnostics: dict[str, Any] = {}

    if model.SolCount > 0:
        objective_value = float(model.ObjVal)
        gap_percent = float(model.MIPGap) * 100.0
        z_truck_value = float(z_truck_expr.getValue())
        z_drone_value = float(z_drone_expr.getValue())

        x_values = {arc: float(x[arc].X) for arc in arc_list}
        route_output = _extract_truck_route(data=data, x_values=x_values)

        order_lookup = {node: idx for idx, node in enumerate(route_output) if node in C}
        truck_customers = [i for i in C if v[i].X > 0.5]
        truck_customers.sort(key=lambda cid: (order_lookup.get(cid, 10**9), cid))

        for k, cand in enumerate(candidates):
            if z[k].X <= 0.5:
                continue
            launch_node = 0 if cand.launch == data.depot_start else cand.launch
            recovery_node = 0 if cand.recovery == data.depot_end else cand.recovery
            sorties_output.append(
                {
                    'launch_node': int(launch_node),
                    'customers': [int(cid) for cid in cand.customers],
                    'recovery_node': int(recovery_node),
                    'energy_wh': _format_float(E[k].X),
                }
            )

        served_set = set(truck_customers)
        for sortie in sorties_output:
            served_set.update(sortie['customers'])

        served_customers = sorted(served_set)
        unserved_customers = [cid for cid in C if cid not in served_set]

        # For drone-served customers, T[i] is not meaningful because truck does not visit i.
        # Diagnostics should expose actual customer service time in both maps.
        service_times_by_customer = {str(i): _format_float(S[i].X) for i in C}
        truck_arrival_times: dict[str, float | None] = {
            str(data.depot_start): _format_float(T[data.depot_start].X),
            str(end): _format_float(T[end].X),
        }
        for i in C:
            if v[i].X > 0.5:
                truck_arrival_times[str(i)] = _format_float(T[i].X)
            else:
                truck_arrival_times[str(i)] = service_times_by_customer[str(i)]

        diagnostics = {
            'truck_arrival_times': truck_arrival_times,
            'service_times': service_times_by_customer,
            'sortie_launch_times': {str(k): _format_float(T_launch[k].X) for k in K if z[k].X > 0.5},
            'sortie_return_times': {str(k): _format_float(T_drone_ret[k].X) for k in K if z[k].X > 0.5},
            'sortie_wait_times': {str(k): _format_float(W[k].X) for k in K if z[k].X > 0.5},
            'sortie_hover_times': {str(k): _format_float(H[k].X) for k in K if z[k].X > 0.5},
            'candidate_count_in_model': len(candidates),
        }

    result = {
        'solver': 'gurobi',
        'instance': data.instance_name,
        'instance_vehicle_pair_count': data.vehicle_pair_count,
        'status': status_text,
        'objective': _format_float(objective_value),
        'gap_percent': _format_float(gap_percent),
        'solve_time_seconds': _format_float(float(model.Runtime)),
        'cost_breakdown': {
            'z_fixed': _format_float(data.fixed_pair_cost),
            'z_truck': _format_float(z_truck_value if z_truck_value is not None else 0.0),
            'z_drone': _format_float(z_drone_value if z_drone_value is not None else 0.0),
        },
        'solution': {
            'truck_route': route_output,
            'truck_customers': truck_customers,
            'sorties': sorties_output,
            'served_customers': len(served_customers),
            'unserved_customers': unserved_customers,
        },
        'diagnostics': diagnostics,
    }

    return result


def write_result_json(result: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    instance_name = str(result['instance'])
    out_path = output_dir / f'{instance_name}_gurobi_result.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return out_path
