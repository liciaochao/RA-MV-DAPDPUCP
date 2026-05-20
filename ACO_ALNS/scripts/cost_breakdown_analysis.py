from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _pick_float(payload: dict[str, Any] | None, keys: list[str]) -> float | None:
    if payload is None:
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _pick_list(payload: dict[str, Any] | None, key: str) -> list[Any]:
    if payload is None:
        return []
    value = payload.get(key)
    if isinstance(value, list):
        return value
    return []


def _resolve_instance_path(case_dir: Path, instance_file: str | None, workspace_root: Path) -> Path | None:
    if not instance_file:
        return None
    candidate = Path(instance_file)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    for path in [
        case_dir / instance_file,
        workspace_root / "examples" / "instances" / instance_file,
        workspace_root / "examples" / instance_file,
    ]:
        if path.exists():
            return path
    return None


def _resolve_config_path(case_dir: Path, config_file: str | None, workspace_root: Path) -> Path | None:
    if not config_file:
        return None
    candidate = Path(config_file)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    for path in [
        case_dir / config_file,
        workspace_root / "examples" / "configs" / config_file,
        workspace_root / "examples" / config_file,
    ]:
        if path.exists():
            return path
    return None


def _extract_customer_xy(instance_payload: dict[str, Any], customer_id: int) -> tuple[float, float] | None:
    customers = instance_payload.get("customers")
    if isinstance(customers, list):
        for item in customers:
            if int(item.get("customer_id", -1)) == customer_id:
                return float(item["x"]), float(item["y"])
    elif isinstance(customers, dict):
        for item in customers.values():
            if int(item.get("customer_id", -1)) == customer_id:
                return float(item["x"]), float(item["y"])
    return None


def _estimate_fail_penalty(
    instance_payload: dict[str, Any] | None,
    config_payload: dict[str, Any] | None,
    failed_customers: list[int],
) -> float | None:
    if instance_payload is None or config_payload is None:
        return None

    cost_payload = config_payload.get("cost")
    if not isinstance(cost_payload, dict):
        return None
    truck_cost = cost_payload.get("truck_cost_per_km")
    if not isinstance(truck_cost, (int, float)):
        return None

    depot = instance_payload.get("depot")
    if not isinstance(depot, dict):
        return None
    depot_x = float(depot.get("x", 0.0))
    depot_y = float(depot.get("y", 0.0))
    truck_scale = float(instance_payload.get("truck_distance_scale", 1.2))

    total = 0.0
    for customer_id in failed_customers:
        coord = _extract_customer_xy(instance_payload, int(customer_id))
        if coord is None:
            continue
        cx, cy = coord
        distance = truck_scale * math.hypot(cx - depot_x, cy - depot_y)
        total += 2.0 * distance * float(truck_cost)
    return total


@dataclass
class CaseRow:
    case: str
    folder: str
    offline_objective: float | None
    online_actual: float | None
    mc_mean: float | None
    mc_deterministic: float | None
    failed_count: int
    fail_penalty_est: float | None
    online_minus_offline: float | None
    mean_minus_offline: float | None

    @property
    def online_lower(self) -> bool:
        return (
            self.online_actual is not None
            and self.offline_objective is not None
            and self.online_actual < self.offline_objective
        )


def _build_case_row(case_dir: Path, workspace_root: Path) -> CaseRow:
    objective_data = _read_json(case_dir / "objective_summary.json")
    mc_data = _read_json(case_dir / "mc_results.json")
    online_data = _read_json(case_dir / "online_execution_report.json")
    solution_detail = _read_json(case_dir / "solution_detail.json")

    offline_objective = _pick_float(objective_data, ["optimized_objective", "initial_objective"])
    mc_mean = _pick_float(mc_data, ["mean", "mean_actual_cost", "avg_cost", "average_actual_cost"])
    mc_deterministic = _pick_float(mc_data, ["deterministic_cost", "deterministic"])
    online_actual = _pick_float(online_data, ["actual_cost"])
    if online_actual is None:
        online_actual = mc_deterministic

    failed_customers_raw = _pick_list(online_data, "failed_customers")
    failed_customers = [int(x) for x in failed_customers_raw if isinstance(x, (int, float, str))]

    case_name = case_dir.name
    instance_file = None
    config_file = None
    if isinstance(solution_detail, dict):
        instance_file_raw = solution_detail.get("instance_file")
        config_file_raw = solution_detail.get("config_file")
        if isinstance(instance_file_raw, str):
            instance_file = instance_file_raw
            case_name = Path(instance_file_raw).stem
        if isinstance(config_file_raw, str):
            config_file = config_file_raw

    instance_payload = _read_json(_resolve_instance_path(case_dir, instance_file, workspace_root) or Path("__missing__"))
    config_payload = _read_json(_resolve_config_path(case_dir, config_file, workspace_root) or Path("__missing__"))
    fail_penalty_est = _estimate_fail_penalty(instance_payload, config_payload, failed_customers)

    online_minus_offline = None
    if online_actual is not None and offline_objective is not None:
        online_minus_offline = online_actual - offline_objective

    mean_minus_offline = None
    if mc_mean is not None and offline_objective is not None:
        mean_minus_offline = mc_mean - offline_objective

    return CaseRow(
        case=case_name,
        folder=str(case_dir),
        offline_objective=offline_objective,
        online_actual=online_actual,
        mc_mean=mc_mean,
        mc_deterministic=mc_deterministic,
        failed_count=len(failed_customers),
        fail_penalty_est=fail_penalty_est,
        online_minus_offline=online_minus_offline,
        mean_minus_offline=mean_minus_offline,
    )


def _discover_case_dirs(roots: list[Path]) -> list[Path]:
    case_dirs: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            files = {p.name for p in path.glob("*.json")}
            if not files:
                continue
            if "objective_summary.json" in files and "mc_results.json" in files:
                if path not in seen:
                    seen.add(path)
                    case_dirs.append(path)
    return sorted(case_dirs)


def analyze_cost_breakdown(roots: list[Path]) -> list[CaseRow]:
    workspace_root = Path(__file__).resolve().parents[1]
    case_dirs = _discover_case_dirs(roots)
    rows = [_build_case_row(case_dir, workspace_root) for case_dir in case_dirs]

    print("=" * 132)
    print("Cost Comparison Report")
    print("=" * 132)
    print(
        f"{'Case':<18} {'OfflineObj':>11} {'OnlineAct':>11} {'MCMean':>11} "
        f"{'Online-Offline':>15} {'MC-Offline':>12} {'Fail#':>6} {'FailPenalty(est)':>17} {'OnlineLower':>12}"
    )
    print("-" * 132)

    lower_count = 0
    for row in rows:
        off = "n/a" if row.offline_objective is None else f"{row.offline_objective:.3f}"
        onl = "n/a" if row.online_actual is None else f"{row.online_actual:.3f}"
        mean = "n/a" if row.mc_mean is None else f"{row.mc_mean:.3f}"
        d1 = "n/a" if row.online_minus_offline is None else f"{row.online_minus_offline:.3f}"
        d2 = "n/a" if row.mean_minus_offline is None else f"{row.mean_minus_offline:.3f}"
        fp = "n/a" if row.fail_penalty_est is None else f"{row.fail_penalty_est:.3f}"
        flag = "YES" if row.online_lower else "NO"
        if row.online_lower:
            lower_count += 1
        print(
            f"{row.case:<18} {off:>11} {onl:>11} {mean:>11} "
            f"{d1:>15} {d2:>12} {row.failed_count:>6} {fp:>17} {flag:>12}"
        )

    print("-" * 132)
    print(f"Online lower than offline objective: {lower_count}/{len(rows)} cases")
    print(f"Scanned roots: {', '.join(str(root) for root in roots)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze offline objective vs online cost from existing result files.")
    parser.add_argument(
        "roots",
        nargs="*",
        default=["results", "outputs"],
        help="Root directories to scan (default: results outputs)",
    )
    args = parser.parse_args()

    roots = [Path(root) for root in args.roots]
    analyze_cost_breakdown(roots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
