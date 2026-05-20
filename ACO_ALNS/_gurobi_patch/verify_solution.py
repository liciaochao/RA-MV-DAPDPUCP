from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_loader import load_problem_data


def _load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _check_route_basic(route: list[int], n_customers: int) -> tuple[bool, str]:
    if not route:
        return False, 'truck_route is empty'
    if route[0] != 0 or route[-1] != 0:
        return False, 'truck_route must start and end at depot 0'

    route_customers = [x for x in route if x != 0]
    if len(set(route_customers)) != len(route_customers):
        return False, 'truck_route contains repeated customer node (potential subcycle/revisit)'
    if len(route_customers) > n_customers:
        return False, 'truck_route has more customer visits than customer count'
    return True, 'route starts/ends at depot and has no repeated customer visits'


def verify_solution(result_path: Path, instance_path: Path, config_path: Path) -> tuple[bool, list[dict[str, Any]]]:
    result = _load_json(result_path)
    data = load_problem_data(instance_path, config_path)

    solution = result.get('solution', {})
    route = list(solution.get('truck_route', []))
    truck_customers = list(solution.get('truck_customers', []))
    sorties = list(solution.get('sorties', []))

    checks: list[dict[str, Any]] = []

    # 1) truck route start/end, no subcycle-like repeats
    ok, msg = _check_route_basic(route, data.n_customers)
    checks.append({'id': 1, 'ok': ok, 'detail': msg})

    # 2) each customer served exactly once
    served_count = {cid: 0 for cid in data.customers}
    for cid in truck_customers:
        if cid in served_count:
            served_count[cid] += 1
    for sortie in sorties:
        for cid in sortie.get('customers', []):
            if cid in served_count:
                served_count[cid] += 1
    bad = [cid for cid, cnt in served_count.items() if cnt != 1]
    checks.append(
        {
            'id': 2,
            'ok': len(bad) == 0,
            'detail': 'all customers served exactly once' if not bad else f'customers with count!=1: {bad}',
        }
    )

    # 3) truck-only customers served by truck
    truck_set = set(truck_customers)
    missing_truck_only = [cid for cid in data.truck_only_customers if cid not in truck_set]
    checks.append(
        {
            'id': 3,
            'ok': len(missing_truck_only) == 0,
            'detail': 'all truck-only customers served by truck' if not missing_truck_only else f'missing: {missing_truck_only}',
        }
    )

    # 4) launch/recovery in truck route
    route_index = {node: idx for idx, node in enumerate(route)}
    bad_launch_recovery: list[str] = []
    for idx, sortie in enumerate(sorties):
        launch = int(sortie.get('launch_node', 0))
        recovery = int(sortie.get('recovery_node', 0))
        if launch != 0 and launch not in route_index:
            bad_launch_recovery.append(f'sortie {idx}: launch {launch} not in route')
        if recovery != 0 and recovery not in route_index:
            bad_launch_recovery.append(f'sortie {idx}: recovery {recovery} not in route')
    checks.append(
        {
            'id': 4,
            'ok': len(bad_launch_recovery) == 0,
            'detail': 'all launch/recovery nodes on truck route' if not bad_launch_recovery else '; '.join(bad_launch_recovery),
        }
    )

    # 5) launch before recovery
    launch_order_violations: list[str] = []
    for idx, sortie in enumerate(sorties):
        launch = int(sortie.get('launch_node', 0))
        recovery = int(sortie.get('recovery_node', 0))
        launch_pos = 0 if launch == 0 else route_index.get(launch)
        recovery_pos = len(route) - 1 if recovery == 0 else route_index.get(recovery)
        if launch_pos is None or recovery_pos is None:
            continue
        if launch_pos >= recovery_pos:
            launch_order_violations.append(
                f'sortie {idx}: launch_pos={launch_pos}, recovery_pos={recovery_pos}'
            )
    checks.append(
        {
            'id': 5,
            'ok': len(launch_order_violations) == 0,
            'detail': 'launch before recovery for all sorties' if not launch_order_violations else '; '.join(launch_order_violations),
        }
    )

    # 6) no crossing intervals
    intervals: list[tuple[int, int, int]] = []
    for idx, sortie in enumerate(sorties):
        launch = int(sortie.get('launch_node', 0))
        recovery = int(sortie.get('recovery_node', 0))
        l_pos = 0 if launch == 0 else route_index.get(launch)
        r_pos = len(route) - 1 if recovery == 0 else route_index.get(recovery)
        if l_pos is None or r_pos is None:
            continue
        intervals.append((idx, l_pos, r_pos))

    crossing: list[str] = []
    for i in range(len(intervals)):
        s1, l1, r1 = intervals[i]
        for j in range(i + 1, len(intervals)):
            s2, l2, r2 = intervals[j]
            cond1 = l1 < l2 < r1 < r2
            cond2 = l2 < l1 < r2 < r1
            if cond1 or cond2:
                crossing.append(f'sortie {s1} vs {s2}')
    checks.append(
        {
            'id': 6,
            'ok': len(crossing) == 0,
            'detail': 'no crossing sortie intervals' if not crossing else '; '.join(crossing),
        }
    )

    # 7) customers per sortie <= max_per_sortie
    too_many = [idx for idx, s in enumerate(sorties) if len(s.get('customers', [])) > data.max_per_sortie]
    checks.append(
        {
            'id': 7,
            'ok': len(too_many) == 0,
            'detail': 'sortie size within cap' if not too_many else f'violations at sorties {too_many}',
        }
    )

    # 8) sortie payload <= drone capacity
    payload_violations: list[str] = []
    for idx, sortie in enumerate(sorties):
        payload = sum(data.weights.get(int(cid), 0.0) for cid in sortie.get('customers', []))
        if payload > data.drone_capacity + 1e-6:
            payload_violations.append(f'sortie {idx}: payload={payload}')
    checks.append(
        {
            'id': 8,
            'ok': len(payload_violations) == 0,
            'detail': 'all sortie payloads within drone capacity' if not payload_violations else '; '.join(payload_violations),
        }
    )

    # 9) sortie energy <= battery
    energy_violations: list[str] = []
    for idx, sortie in enumerate(sorties):
        energy = float(sortie.get('energy_wh', 0.0))
        if energy > data.battery_capacity + 1e-6:
            energy_violations.append(f'sortie {idx}: energy={energy}')
    checks.append(
        {
            'id': 9,
            'ok': len(energy_violations) == 0,
            'detail': 'all sortie energies within battery cap' if not energy_violations else '; '.join(energy_violations),
        }
    )

    # 10) truck load <= capacity (conservative sum check)
    truck_load = sum(data.weights.get(int(cid), 0.0) for cid in truck_customers)
    checks.append(
        {
            'id': 10,
            'ok': truck_load <= data.truck_capacity + 1e-6,
            'detail': f'truck total assigned load={truck_load}, cap={data.truck_capacity}',
        }
    )

    diagnostics = result.get('diagnostics', {})

    # 11) time windows
    service_times = diagnostics.get('service_times', {})
    tw_violations: list[str] = []
    if service_times:
        for cid in data.customers:
            key = str(cid)
            if key not in service_times:
                tw_violations.append(f'customer {cid}: missing service time')
                continue
            st = float(service_times[key])
            lb, ub = data.time_windows[cid]
            if st < lb - 1e-6 or st > ub + 1e-6:
                tw_violations.append(f'customer {cid}: t={st}, tw=[{lb},{ub}]')
    else:
        tw_violations.append('missing diagnostics.service_times in result json')

    checks.append(
        {
            'id': 11,
            'ok': len(tw_violations) == 0,
            'detail': 'all service times satisfy time windows' if not tw_violations else '; '.join(tw_violations),
        }
    )

    # 12) truck wait <= max_wait_time
    wait_times = diagnostics.get('sortie_wait_times', {})
    wait_violations: list[str] = []
    if wait_times:
        for sid, wait in wait_times.items():
            if float(wait) > data.max_wait_time + 1e-6:
                wait_violations.append(f'sortie {sid}: wait={wait}')
    checks.append(
        {
            'id': 12,
            'ok': len(wait_violations) == 0,
            'detail': 'all waits within max_wait_time' if not wait_violations else '; '.join(wait_violations),
        }
    )

    all_ok = all(item['ok'] for item in checks)
    return all_ok, checks


def main() -> int:
    parser = argparse.ArgumentParser(description='Verify feasibility of gurobi result json.')
    parser.add_argument('--result', required=True, type=str)
    parser.add_argument('--instance', required=True, type=str)
    parser.add_argument('--config', required=True, type=str)
    args = parser.parse_args()

    all_ok, checks = verify_solution(
        result_path=Path(args.result).resolve(),
        instance_path=Path(args.instance).resolve(),
        config_path=Path(args.config).resolve(),
    )

    print('Verification summary:')
    for item in checks:
        status = 'PASS' if item['ok'] else 'FAIL'
        print(f"  [{status}] #{item['id']}: {item['detail']}")

    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
