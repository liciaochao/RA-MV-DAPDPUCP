from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from copy import deepcopy
from pathlib import Path


REPRESENTATIVE_MEDIUM_INSTANCES = [
    "medium_40c_3v_seed1.json",
    "medium_50c_3v_seed1.json",
    "medium_60c_4v_seed1.json",
    "medium_80c_5v_seed1.json",
    "medium_100c_7v_seed1.json",
]


P0_GRID_PROFILES = [
    {
        "name": "p0_m_balanced",
        "aco": {},
    },
    {
        "name": "p0_m_tw_focus",
        "aco": {
            "eta_weight_distance": 1.0,
            "eta_weight_time_window": 1.2,
            "eta_weight_capacity": 0.6,
            "candidate_list_size": 16,
            "global_best_weight": 0.55,
        },
    },
    {
        "name": "p0_m_capacity_focus",
        "aco": {
            "eta_weight_distance": 1.0,
            "eta_weight_time_window": 0.8,
            "eta_weight_capacity": 1.1,
            "candidate_list_size": 16,
            "global_best_weight": 0.55,
        },
    },
    {
        "name": "p0_m_explore",
        "aco": {
            "candidate_list_size": 24,
            "evaporation_rate": 0.25,
            "global_best_weight": 0.4,
            "stagnation_restart_threshold": 8,
            "stagnation_restart_ratio": 0.4,
            "ant_count": 6,
        },
    },
    {
        "name": "p0_m_exploit",
        "aco": {
            "candidate_list_size": 12,
            "evaporation_rate": 0.15,
            "global_best_weight": 0.8,
            "stagnation_restart_threshold": 14,
            "ant_count": 5,
            "max_iterations": 18,
        },
    },
    {
        "name": "p0_m_restart_aggressive",
        "aco": {
            "candidate_list_size": 18,
            "stagnation_restart_threshold": 6,
            "stagnation_restart_ratio": 0.5,
            "tau_min": 0.05,
            "tau_max": 8.0,
            "global_best_weight": 0.5,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run P0 parameter grid on representative medium instances."
    )
    parser.add_argument(
        "--base-config",
        default="examples/configs/config_pure_aco_medium.json",
        help="Base pure ACO config used to generate profile configs.",
    )
    parser.add_argument(
        "--instances-dir",
        default="examples/instances",
        help="Directory for instance json files.",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        default=REPRESENTATIVE_MEDIUM_INSTANCES,
        help="Instance file names (or paths).",
    )
    parser.add_argument(
        "--results-root",
        default="results/p0_medium_grid",
        help="Root directory for generated configs and run outputs.",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=[],
        help="Optional subset of profile names to run.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute runs. Without this flag, only dry-run planning is performed.",
    )
    return parser.parse_args()


def _resolve(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def _load_instances(project_root: Path, instances_dir: Path, raw_instances: list[str]) -> list[Path]:
    _ = project_root
    resolved: list[Path] = []
    for item in raw_instances:
        item_path = Path(item)
        if item_path.is_absolute():
            path = item_path
        elif item_path.suffix.lower() == ".json":
            path = instances_dir / item_path
        else:
            path = instances_dir / f"{item}.json"
        if not path.exists():
            raise FileNotFoundError(f"instance not found: {path}")
        resolved.append(path.resolve())
    if not resolved:
        raise ValueError("instances list is empty")
    return resolved


def _select_profiles(names: list[str]) -> list[dict[str, object]]:
    if not names:
        return list(P0_GRID_PROFILES)
    wanted = set(names)
    selected = [profile for profile in P0_GRID_PROFILES if profile["name"] in wanted]
    missing = sorted(wanted - {profile["name"] for profile in selected})
    if missing:
        raise ValueError(f"unknown profile(s): {', '.join(missing)}")
    return selected


def _prepare_profile_config(
    base_config: dict[str, object],
    profile: dict[str, object],
) -> dict[str, object]:
    config = deepcopy(base_config)
    config["deterministic"] = True
    config["optimizer"] = {"type": "pure_aco"}

    alns = dict(config.get("alns", {}))
    alns["iterations"] = 0
    config["alns"] = alns

    aco = dict(config.get("aco", {}))
    aco.update(profile["aco"])
    config["aco"] = aco
    return config


def _write_profile_configs(
    base_config_path: Path,
    results_root: Path,
    profiles: list[dict[str, object]],
    instances: list[Path],
) -> list[dict[str, object]]:
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    config_dir = results_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for profile in profiles:
        profile_name = str(profile["name"])
        config_payload = _prepare_profile_config(base_config, profile)
        out_path = config_dir / f"{profile_name}.json"
        out_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        records.append(
            {
                "profile": profile_name,
                "config_path": str(out_path.resolve()),
                "aco_overrides": dict(profile["aco"]),
                "instances": [str(path) for path in instances],
            }
        )

    manifest_path = results_root / "p0_grid_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "base_config": str(base_config_path.resolve()),
                "profile_count": len(records),
                "instances": [str(path) for path in instances],
                "profiles": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Wrote manifest: {manifest_path}")
    return records


def _summarize_run(summary_path: Path) -> str:
    if not summary_path.exists():
        return "no summary generated"

    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if not rows:
        return "empty summary"

    ok_count = sum(1 for row in rows if row.get("status") == "OK")
    elapsed = [float(row.get("elapsed_seconds", 0.0)) for row in rows]
    mean_elapsed = statistics.mean(elapsed) if elapsed else 0.0
    return f"ok={ok_count}/{len(rows)}, mean_time={mean_elapsed:.1f}s"


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    base_config_path = _resolve(project_root, args.base_config)
    if not base_config_path.exists():
        raise FileNotFoundError(f"base config not found: {base_config_path}")

    instances_dir = _resolve(project_root, args.instances_dir)
    if not instances_dir.exists():
        raise FileNotFoundError(f"instances dir not found: {instances_dir}")

    results_root = _resolve(project_root, args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    instances = _load_instances(project_root, instances_dir, list(args.instances))
    profiles = _select_profiles(list(args.profiles))
    records = _write_profile_configs(
        base_config_path=base_config_path,
        results_root=results_root,
        profiles=profiles,
        instances=instances,
    )

    run_mode = "EXECUTE" if args.execute else "DRY-RUN"
    print(f"[INFO] Mode: {run_mode}")
    print(f"[INFO] Profiles: {', '.join(record['profile'] for record in records)}")
    print(f"[INFO] Instances: {', '.join(path.name for path in instances)}")

    for idx, record in enumerate(records, 1):
        profile = str(record["profile"])
        config_path = Path(str(record["config_path"]))
        profile_results_dir = results_root / "runs" / profile
        profile_results_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "run_experiments.py",
            "--config",
            str(config_path),
            "--instances",
            *[str(path) for path in instances],
            "--results-dir",
            str(profile_results_dir),
            "--no-plot",
        ]
        if not args.execute:
            cmd.append("--dry-run")

        print(f"\n[{idx}/{len(records)}] profile={profile}")
        print(" ".join(cmd))
        completed = subprocess.run(cmd, cwd=str(project_root), check=False)
        if completed.returncode != 0:
            print(f"[WARN] profile {profile} failed with code {completed.returncode}")
            continue

        summary_line = _summarize_run(profile_results_dir / "experiment_summary.json")
        print(f"[INFO] {profile}: {summary_line}")

    print(f"\n[INFO] Grid output root: {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
