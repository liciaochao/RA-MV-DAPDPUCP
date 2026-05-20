from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


SCALE_SETTINGS: dict[str, dict[str, object]] = {
    "small": {
        "base_config": "examples/configs/config_pure_aco_small.json",
        "target_configs": [
            "examples/configs/config_pure_aco_small.json",
            "examples/configs/config_small.json",
        ],
        "instances": [
            "small_10c_1v_seed1.json",
            "small_16c_1v_seed1.json",
            "small_20c_2v_seed1.json",
        ],
        "profiles": [
            {"name": "balanced", "aco": {}},
            {
                "name": "constraint_focus",
                "aco": {
                    "eta_weight_time_window": 1.0,
                    "eta_weight_capacity": 0.8,
                    "candidate_list_size": 10,
                    "global_best_weight": 0.65,
                },
            },
            {
                "name": "explore_restart",
                "aco": {
                    "candidate_list_size": 18,
                    "evaporation_rate": 0.25,
                    "global_best_weight": 0.45,
                    "stagnation_restart_threshold": 10,
                    "stagnation_restart_ratio": 0.4,
                },
            },
        ],
    },
    "medium": {
        "base_config": "examples/configs/config_pure_aco_medium.json",
        "target_configs": [
            "examples/configs/config_pure_aco_medium.json",
            "examples/configs/config_medium.json",
        ],
        "instances": [
            "medium_30c_2v_seed1.json",
            "medium_60c_4v_seed1.json",
            "medium_100c_7v_seed1.json",
        ],
        "profiles": [
            {"name": "balanced", "aco": {}},
            {
                "name": "constraint_focus",
                "aco": {
                    "eta_weight_time_window": 1.1,
                    "eta_weight_capacity": 0.9,
                    "candidate_list_size": 16,
                    "global_best_weight": 0.65,
                },
            },
            {
                "name": "explore_restart",
                "aco": {
                    "candidate_list_size": 24,
                    "evaporation_rate": 0.25,
                    "global_best_weight": 0.45,
                    "stagnation_restart_threshold": 8,
                    "stagnation_restart_ratio": 0.4,
                },
            },
        ],
    },
    "large": {
        "base_config": "examples/configs/config_pure_aco_large.json",
        "target_configs": [
            "examples/configs/config_pure_aco_large.json",
            "examples/configs/config_large.json",
        ],
        "instances": [
            "large_150c_9v_seed1.json",
            "large_250c_15v_seed1.json",
        ],
        "profiles": [
            {"name": "balanced", "aco": {}},
            {
                "name": "constraint_focus",
                "aco": {
                    "eta_weight_time_window": 1.15,
                    "eta_weight_capacity": 1.0,
                    "candidate_list_size": 20,
                    "global_best_weight": 0.65,
                    "tau_max": 8.0,
                },
            },
            {
                "name": "explore_restart",
                "aco": {
                    "candidate_list_size": 28,
                    "evaporation_rate": 0.25,
                    "global_best_weight": 0.45,
                    "stagnation_restart_threshold": 6,
                    "stagnation_restart_ratio": 0.45,
                    "ant_count": 6,
                },
            },
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune P0 ACO profiles for small/medium/large and update winner configs."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root path.",
    )
    parser.add_argument(
        "--results-root",
        default="results/p0_tuning",
        help="Directory for tuning outputs.",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        choices=["small", "medium", "large"],
        default=["small", "medium", "large"],
        help="Scales to tune.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and commands only, do not execute.",
    )
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="Do not write winner values back to config files.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="If summary exists, skip rerun and reuse existing experiment_summary.json.",
    )
    return parser.parse_args()


def _resolve(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prepare_profile_config(base_payload: dict[str, object], profile_overrides: dict[str, object]) -> dict[str, object]:
    payload = json.loads(json.dumps(base_payload, ensure_ascii=False))
    payload["deterministic"] = True
    payload["optimizer"] = {"type": "pure_aco"}

    alns = dict(payload.get("alns", {}))
    alns["iterations"] = 0
    payload["alns"] = alns

    aco = dict(payload.get("aco", {}))
    aco.update(profile_overrides)
    payload["aco"] = aco
    return payload


def _run_command(cmd: list[str], cwd: Path) -> int:
    completed = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(completed.returncode)


def _load_summary(summary_path: Path) -> list[dict[str, object]]:
    if not summary_path.exists():
        return []
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else []


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _profile_metrics(rows: list[dict[str, object]]) -> dict[str, float | int | None]:
    total = len(rows)
    ok_rows = [row for row in rows if str(row.get("status")) == "OK"]

    objectives = [
        _safe_float(row.get("optimized_objective"))
        for row in ok_rows
        if _safe_float(row.get("optimized_objective")) is not None
    ]
    times = [
        _safe_float(row.get("elapsed_seconds"))
        for row in ok_rows
        if _safe_float(row.get("elapsed_seconds")) is not None
    ]

    success_rate = (len(ok_rows) / total) if total > 0 else 0.0
    mean_obj = statistics.mean(objectives) if objectives else math.inf
    std_obj = statistics.pstdev(objectives) if len(objectives) > 1 else 0.0
    cv_obj = (std_obj / abs(mean_obj)) if objectives and abs(mean_obj) > 1e-12 else math.inf
    mean_time = statistics.mean(times) if times else math.inf

    return {
        "total_cases": total,
        "ok_cases": len(ok_rows),
        "success_rate": success_rate,
        "mean_objective": mean_obj,
        "std_objective": std_obj,
        "cv_objective": cv_obj,
        "mean_time_seconds": mean_time,
    }


def _minmax_normalize(value: float, min_v: float, max_v: float) -> float:
    if not math.isfinite(value):
        return 1.0
    if max_v - min_v <= 1e-12:
        return 0.0
    return (value - min_v) / (max_v - min_v)


def _score_profiles(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    mean_objs = [float(row["metrics"]["mean_objective"]) for row in metric_rows]
    cvs = [float(row["metrics"]["cv_objective"]) for row in metric_rows]
    times = [float(row["metrics"]["mean_time_seconds"]) for row in metric_rows]

    obj_min, obj_max = min(mean_objs), max(mean_objs)
    cv_min, cv_max = min(cvs), max(cvs)
    time_min, time_max = min(times), max(times)

    for row in metric_rows:
        metrics = row["metrics"]
        success_rate = float(metrics["success_rate"])
        obj_norm = _minmax_normalize(float(metrics["mean_objective"]), obj_min, obj_max)
        cv_norm = _minmax_normalize(float(metrics["cv_objective"]), cv_min, cv_max)
        time_norm = _minmax_normalize(float(metrics["mean_time_seconds"]), time_min, time_max)
        fail_penalty = (1.0 - success_rate) * 5.0
        score = 0.60 * obj_norm + 0.25 * cv_norm + 0.15 * time_norm + fail_penalty
        row["score"] = score
        row["score_detail"] = {
            "objective_norm": obj_norm,
            "stability_norm": cv_norm,
            "time_norm": time_norm,
            "failure_penalty": fail_penalty,
        }

    return sorted(metric_rows, key=lambda item: float(item["score"]))


def _apply_winner_to_configs(project_root: Path, target_configs: list[str], winner_overrides: dict[str, object]) -> list[str]:
    updated: list[str] = []
    for config_rel in target_configs:
        config_path = _resolve(project_root, config_rel)
        payload = _load_json(config_path)
        aco = dict(payload.get("aco", {}))
        aco.update(winner_overrides)
        payload["aco"] = aco
        _save_json(config_path, payload)
        updated.append(str(config_path))
    return updated


def main() -> int:
    args = parse_args()
    project_root = _resolve(Path.cwd(), args.project_root)
    results_root = _resolve(project_root, args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    tuning_report: dict[str, object] = {
        "results_root": str(results_root),
        "scales": {},
    }

    for scale in args.scales:
        setting = SCALE_SETTINGS[scale]
        base_config_path = _resolve(project_root, str(setting["base_config"]))
        base_payload = _load_json(base_config_path)
        scale_dir = results_root / scale
        config_dir = scale_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        instances = [_resolve(project_root, "examples/instances" + "/" + name) for name in list(setting["instances"])]
        profile_rows: list[dict[str, object]] = []

        print(f"\n===== TUNING SCALE: {scale} =====")
        for profile in list(setting["profiles"]):
            profile_name = str(profile["name"])
            overrides = dict(profile["aco"])
            profile_config = _prepare_profile_config(base_payload, overrides)
            profile_config_path = config_dir / f"{profile_name}.json"
            _save_json(profile_config_path, profile_config)

            profile_results_dir = scale_dir / "runs" / profile_name
            profile_results_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                "run_experiments.py",
                "--config",
                str(profile_config_path),
                "--instances",
                *[str(path) for path in instances],
                "--results-dir",
                str(profile_results_dir),
                "--no-plot",
            ]
            if args.dry_run:
                cmd.append("--dry-run")

            print(f"[{scale}] profile={profile_name}")
            summary_path = profile_results_dir / "experiment_summary.json"
            if args.reuse_existing and summary_path.exists() and not args.dry_run:
                print(f"[REUSE] {summary_path}")
                rc = 0
            else:
                print(" ".join(cmd))
                rc = _run_command(cmd, project_root)
            rows = _load_summary(summary_path)
            metrics = _profile_metrics(rows)

            profile_rows.append(
                {
                    "profile": profile_name,
                    "overrides": overrides,
                    "return_code": rc,
                    "summary_path": str(summary_path),
                    "metrics": metrics,
                }
            )

        ranked = _score_profiles(profile_rows)
        winner = ranked[0] if ranked else None

        scale_report: dict[str, object] = {
            "base_config": str(base_config_path),
            "instances": [str(path) for path in instances],
            "ranked_profiles": ranked,
            "winner": winner,
        }

        if winner is not None and not args.no_apply and not args.dry_run:
            updated_files = _apply_winner_to_configs(
                project_root=project_root,
                target_configs=list(setting["target_configs"]),
                winner_overrides=dict(winner["overrides"]),
            )
            scale_report["updated_configs"] = updated_files
            print(f"[APPLY] {scale} winner={winner['profile']} -> {updated_files}")
        else:
            scale_report["updated_configs"] = []

        tuning_report["scales"][scale] = scale_report

    report_path = results_root / "tuning_report.json"
    _save_json(report_path, tuning_report)

    print("\n===== TUNING DONE =====")
    print(f"Report: {report_path}")
    for scale in args.scales:
        winner = tuning_report["scales"][scale]["winner"]
        if winner is None:
            print(f"{scale}: no winner")
            continue
        print(
            f"{scale}: winner={winner['profile']} "
            f"mean_obj={winner['metrics']['mean_objective']:.4f} "
            f"cv={winner['metrics']['cv_objective']:.6f} "
            f"time={winner['metrics']['mean_time_seconds']:.2f}s "
            f"score={winner['score']:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
