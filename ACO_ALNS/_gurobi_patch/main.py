from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from data_loader import load_problem_data, resolve_existing_path
from milp_solver import check_gurobi_available, solve_deterministic_milp, write_result_json
from sortie_enumerator import enumerate_sorties


def _collect_instance_paths(instance: str | None, batch: str | None) -> list[Path]:
    if bool(instance) == bool(batch):
        raise ValueError('Use exactly one of --instance or --batch.')

    if instance:
        return [resolve_existing_path(instance)]

    batch_path = resolve_existing_path(batch)
    if batch_path.is_file():
        return [batch_path]

    paths = sorted(p for p in batch_path.glob('*.json') if p.is_file())
    if not paths:
        raise ValueError(f'No json instances found in: {batch_path}')
    return paths


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _max_from_mapping(mapping: Any, default: float = 0.0) -> float:
    if not isinstance(mapping, dict) or not mapping:
        return default
    vals: list[float] = []
    for value in mapping.values():
        try:
            vals.append(float(value))
        except Exception:
            continue
    return max(vals) if vals else default


def _resolve_candidate_cap(n_customers: int, override: int | None) -> int | None:
    if override is not None:
        return override
    if n_customers <= 15:
        return None
    if n_customers <= 20:
        return 200
    return 100


def _resolve_pair_limit(n_customers: int, override: int | None) -> int | None:
    if override is not None:
        return override
    if n_customers <= 15:
        return None
    return 10


def _candidate_cap_fallbacks(base_cap: int | None, n_customers: int) -> list[int | None]:
    caps: list[int | None] = [base_cap]

    if n_customers <= 15:
        fallbacks = [600, 500, 400, 300, 200, 150, 100]
    elif n_customers <= 20:
        fallbacks = [200, 180, 150, 120, 100, 80]
    else:
        fallbacks = [100, 80, 60, 50]

    for cap in fallbacks:
        if base_cap is None:
            if cap not in caps:
                caps.append(cap)
        else:
            if cap < base_cap and cap not in caps:
                caps.append(cap)
    return caps


def _write_summary_csv(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'comparison_summary.csv'
    headers = [
        'instance',
        'gurobi_obj',
        'gurobi_gap_pct',
        'gurobi_time_s',
        'gurobi_status',
        'z_fixed',
        'z_truck',
        'z_drone',
        'n_sorties',
        'n_candidates_feasible',
        'n_candidates_selected',
        'max_arrival_time',
        'max_wait_time',
    ]

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def run(args: argparse.Namespace) -> int:
    ok, version_or_error = check_gurobi_available()
    if not ok:
        print(f'[ERROR] Gurobi unavailable: {version_or_error}', file=sys.stderr)
        return 2

    print(f'[INFO] Gurobi version: {version_or_error}')

    instance_paths = _collect_instance_paths(args.instance, args.batch)
    config_path = resolve_existing_path(args.config)
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []

    for instance_path in instance_paths:
        print(f'\n[INFO] Solving instance: {instance_path.name}')
        data = load_problem_data(instance_path, config_path)
        for warn in data.warnings:
            print(f'[WARN] {warn}')

        n_customers = len(data.customers)
        pair_limit = _resolve_pair_limit(n_customers, args.pair_limit_per_order)
        base_cap = _resolve_candidate_cap(n_customers, args.candidate_cap)
        candidate_caps = _candidate_cap_fallbacks(base_cap, n_customers)

        result = None
        enum_stats = None
        used_cap = None

        for cap in candidate_caps:
            candidates, enum_stats = enumerate_sorties(
                data,
                pair_limit_per_order=pair_limit,
                model_candidate_cap=cap,
                min_candidates_per_customer=args.min_candidates_per_customer,
            )
            print(
                '[INFO] Sortie enumeration: '
                f'customers={n_customers}, '
                f'feasible={enum_stats.generated_feasible}, '
                f'selected={enum_stats.selected_for_model}, '
                f'pair_limit={enum_stats.pair_limit_per_order}, '
                f'cap={enum_stats.global_cap}'
            )

            try:
                result = solve_deterministic_milp(
                    data=data,
                    candidates=candidates,
                    time_limit=args.time_limit,
                    mip_gap=args.mip_gap,
                    threads=args.threads,
                    output_flag=args.output_flag,
                )
                used_cap = cap
                break
            except Exception as exc:
                message = str(exc).lower()
                if 'size-limited license' in message and cap != candidate_caps[-1]:
                    print(
                        '[WARN] Model exceeds size-limited license with '
                        f'candidate_cap={cap}; retrying with smaller cap.'
                    )
                    continue
                raise

        if result is None or enum_stats is None:
            raise RuntimeError('Solver did not return a result.')

        result['enumeration'] = {
            'generated_feasible': enum_stats.generated_feasible,
            'selected_for_model': enum_stats.selected_for_model,
            'pair_limit_per_order': enum_stats.pair_limit_per_order,
            'global_cap': enum_stats.global_cap,
            'candidate_cap_used': used_cap,
        }

        out_path = write_result_json(result, output_dir)
        print(f'[INFO] Result JSON: {out_path}')
        print(
            '[INFO] Solve summary: '
            f"status={result['status']}, "
            f"obj={result['objective']}, "
            f"gap%={result['gap_percent']}, "
            f"time_s={result['solve_time_seconds']}"
        )

        diagnostics = result.get('diagnostics', {})
        summary_rows.append(
            {
                'instance': result['instance'],
                'gurobi_obj': result['objective'] if result['objective'] is not None else '',
                'gurobi_gap_pct': result['gap_percent'] if result['gap_percent'] is not None else '',
                'gurobi_time_s': result['solve_time_seconds'] if result['solve_time_seconds'] is not None else '',
                'gurobi_status': result['status'],
                'z_fixed': _to_float(result['cost_breakdown']['z_fixed']),
                'z_truck': _to_float(result['cost_breakdown']['z_truck']),
                'z_drone': _to_float(result['cost_breakdown']['z_drone']),
                'n_sorties': len(result.get('solution', {}).get('sorties', [])),
                'n_candidates_feasible': result.get('enumeration', {}).get('generated_feasible', ''),
                'n_candidates_selected': result.get('enumeration', {}).get('selected_for_model', ''),
                'max_arrival_time': _max_from_mapping(diagnostics.get('truck_arrival_times', {}), default=0.0),
                'max_wait_time': _max_from_mapping(diagnostics.get('sortie_wait_times', {}), default=0.0),
            }
        )

    csv_path = _write_summary_csv(summary_rows, output_dir)
    print(f'\n[INFO] Comparison CSV: {csv_path}')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Deterministic truck-drone MILP solver (experiment 1).')
    parser.add_argument('--instance', type=str, default=None, help='Single instance json path.')
    parser.add_argument('--batch', type=str, default=None, help='Directory containing instance json files.')
    parser.add_argument('--config', type=str, required=True, help='Config json path.')
    parser.add_argument('--output', type=str, required=True, help='Output directory for result json/csv.')

    parser.add_argument(
        '--candidate-cap',
        type=int,
        default=None,
        help='Optional max sortie candidates kept in MILP. Default is dynamic by instance size.',
    )
    parser.add_argument(
        '--pair-limit-per-order',
        type=int,
        default=None,
        help='Optional per-ordered-customer cap for launch/recovery alternatives. Default is dynamic.',
    )
    parser.add_argument(
        '--min-candidates-per-customer',
        type=int,
        default=8,
        help='Minimum retained candidates per drone-eligible customer before global fill.',
    )

    parser.add_argument('--time-limit', type=int, default=3600)
    parser.add_argument('--mip-gap', type=float, default=0.005)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--output-flag', type=int, default=1)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    ns = parser.parse_args()
    raise SystemExit(run(ns))
