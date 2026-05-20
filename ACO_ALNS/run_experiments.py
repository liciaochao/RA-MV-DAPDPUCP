import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def create_temp_config(base_config_path: Path, k_cluster: int, temp_path: Path) -> None:
    config = json.loads(base_config_path.read_text(encoding="utf-8"))
    config["etprc"]["k_cluster"] = int(k_cluster)
    temp_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SPD experiments")
    parser.add_argument("--scale", choices=["small", "medium", "large"], help="只运行指定规模")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    parser.add_argument("--fast", action="store_true", help="使用 fast 配置快速验证")
    parser.add_argument("--no-plot", action="store_true", help="传递给 spd.main，跳过可视化绘图")
    parser.add_argument("--config", type=str, default=None, help="指定统一配置文件，覆盖按规模自动选择")
    parser.add_argument("--instances-dir", type=str, default=None, help="指定算例目录，默认 examples/instances")
    parser.add_argument("--instances", nargs="+", default=None, help="显式指定算例文件列表")
    parser.add_argument("--results-dir", type=str, default=None, help="实验结果目录，默认 results/")
    return parser.parse_args()


def _extract_scale(instance_name: str) -> str | None:
    if instance_name.startswith("small_"):
        return "small"
    if instance_name.startswith("medium_"):
        return "medium"
    if instance_name.startswith("large_"):
        return "large"
    return None


def _extract_json_line(stdout_text: str) -> str:
    lines = [line.strip() for line in stdout_text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("{") and line.endswith("}"):
            return line
    return lines[-1] if lines else ""


def _parse_json_line(json_line: str) -> dict[str, object]:
    if not json_line:
        return {}
    try:
        parsed = json.loads(json_line)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _build_subprocess_env(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(project_root / "src")
    prev = env.get("PYTHONPATH", "")
    if prev:
        env["PYTHONPATH"] = src_path + os.pathsep + prev
    else:
        env["PYTHONPATH"] = src_path
    return env


def _build_plan(
    instance_files: list[Path],
    scale_config_map: dict[str, Path],
    results_dir: Path,
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for inst_path in instance_files:
        inst_name = inst_path.stem
        scale = _extract_scale(inst_name)
        if scale is None:
            plan.append(
                {
                    "instance_name": inst_name,
                    "instance_path": inst_path,
                    "scale": "unknown",
                    "base_config": None,
                    "result_dir": results_dir / inst_name,
                    "k_cluster": None,
                    "customers": None,
                }
            )
            continue

        inst_data = json.loads(inst_path.read_text(encoding="utf-8"))
        k_cluster = int(inst_data["vehicle_pair_count"])
        plan.append(
            {
                "instance_name": inst_name,
                "instance_path": inst_path,
                "scale": scale,
                "base_config": scale_config_map[scale],
                "result_dir": results_dir / inst_name,
                "k_cluster": k_cluster,
                "customers": len(inst_data.get("customers", [])),
            }
        )
    return plan


def _resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def run_all_experiments() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    instances_dir = _resolve_path(project_root, args.instances_dir) if args.instances_dir else (project_root / "examples" / "instances")
    configs_dir = project_root / "examples" / "configs"
    results_dir = _resolve_path(project_root, args.results_dir) if args.results_dir else (project_root / "results")
    temp_config = project_root / "_temp_config.json"

    if not instances_dir.exists():
        raise FileNotFoundError(f"instances dir not found: {instances_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 支持命令行显式传入算例列表，便于小批量快速验证。
    if args.instances:
        instance_files = [_resolve_path(project_root, raw_path) for raw_path in args.instances]
        for inst_path in instance_files:
            if not inst_path.exists():
                raise FileNotFoundError(f"instance file not found: {inst_path}")
    else:
        instance_files = sorted(instances_dir.glob("*.json"))
        if args.scale:
            instance_files = [p for p in instance_files if p.stem.startswith(f"{args.scale}_")]

    if args.config:
        config_override = _resolve_path(project_root, args.config)
        if not config_override.exists():
            raise FileNotFoundError(f"config file not found: {config_override}")
        scale_config_map = {
            "small": config_override,
            "medium": config_override,
            "large": config_override,
        }
    elif args.fast:
        fast_config = configs_dir / "config_fast.json"
        scale_config_map = {
            "small": fast_config,
            "medium": fast_config,
            "large": fast_config,
        }
    else:
        scale_config_map = {
            "small": configs_dir / "config_small.json",
            "medium": configs_dir / "config_medium.json",
            "large": configs_dir / "config_large.json",
        }

    plan = _build_plan(instance_files, scale_config_map, results_dir)
    total = len(plan)
    results_summary: list[dict[str, object]] = []
    run_env = _build_subprocess_env(project_root)

    print("===== EXPERIMENT PLAN =====")
    print(f"Project root: {project_root}")
    print(f"Instances dir: {instances_dir}")
    print(f"Results dir: {results_dir}")
    print(f"Total planned: {total}")
    print(f"Mode: {'DRY-RUN' if args.dry_run else 'EXECUTE'}")
    print(f"Fast config: {args.fast}")
    print(f"No plot: {args.no_plot}")
    if args.config:
        print(f"Config override: {args.config}")
    if args.instances:
        print(f"Instances override count: {len(args.instances)}")
    if args.scale:
        print(f"Scale filter: {args.scale}")

    try:
        for idx, item in enumerate(plan, 1):
            inst_name = str(item["instance_name"])
            scale = str(item["scale"])
            base_config = item["base_config"]
            result_dir = Path(item["result_dir"])
            k_cluster = item["k_cluster"]
            customers = item["customers"]

            if scale == "unknown" or base_config is None or k_cluster is None:
                print(f"[{idx}/{total}] [SKIP] {inst_name}: unknown scale")
                results_summary.append(
                    {
                        "instance": inst_name,
                        "scale": scale,
                        "customers": customers,
                        "vehicle_pairs": k_cluster,
                        "status": "SKIP",
                        "elapsed_seconds": 0.0,
                        "optimized_objective": None,
                        "solve_seconds": None,
                    }
                )
                continue

            print(
                f"\n[{idx}/{total}] {inst_name} | scale={scale} | config={Path(base_config).name} | "
                f"k_cluster={k_cluster} | result={result_dir}"
            )

            if args.dry_run:
                print("  [DRY-RUN] skipped execution")
                results_summary.append(
                    {
                        "instance": inst_name,
                        "scale": scale,
                        "customers": customers,
                        "vehicle_pairs": k_cluster,
                        "status": "DRY_RUN",
                        "elapsed_seconds": 0.0,
                        "optimized_objective": None,
                        "solve_seconds": None,
                    }
                )
                continue

            create_temp_config(Path(base_config), int(k_cluster), temp_config)
            result_dir.mkdir(parents=True, exist_ok=True)

            start_time = time.perf_counter()
            cmd = [
                sys.executable,
                "-m",
                "spd.main",
                "solve",
                "--instance",
                str(item["instance_path"]),
                "--config",
                str(temp_config),
                "--output-dir",
                str(result_dir),
            ]
            if args.no_plot:
                cmd.append("--no-plot")

            status = "FAILED"
            elapsed = 0.0
            objective_value: float | None = None
            solve_seconds: float | None = None
            try:
                run_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=720000,
                    cwd=str(project_root),
                    env=run_env,
                )
                elapsed = time.perf_counter() - start_time

                if run_result.returncode == 0:
                    status = "OK"
                    tail = _extract_json_line(run_result.stdout)
                    parsed = _parse_json_line(tail)
                    if "optimized_objective" in parsed:
                        try:
                            objective_value = float(parsed["optimized_objective"])
                        except (TypeError, ValueError):
                            objective_value = None
                    if "solve_seconds" in parsed:
                        try:
                            solve_seconds = float(parsed["solve_seconds"])
                        except (TypeError, ValueError):
                            solve_seconds = None
                    print(f"  [OK] {elapsed:.1f}s | {tail}")
                else:
                    status = "FAILED"
                    print(f"  [FAILED] {elapsed:.1f}s")
                    stderr_preview = run_result.stderr.strip()[:500]
                    stdout_preview = run_result.stdout.strip()[:500]
                    if stderr_preview:
                        print(f"  stderr: {stderr_preview}")
                    if stdout_preview:
                        print(f"  stdout: {stdout_preview}")
            except subprocess.TimeoutExpired:
                elapsed = 7200.0
                status = "TIMEOUT"
                print(f"  [TIMEOUT] {elapsed:.1f}s")
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                status = f"ERROR: {exc}"
                print(f"  [ERROR] {elapsed:.1f}s | {exc}")

            results_summary.append(
                {
                    "instance": inst_name,
                    "scale": scale,
                    "customers": customers,
                    "vehicle_pairs": k_cluster,
                    "status": status,
                    "elapsed_seconds": round(elapsed, 1),
                    "optimized_objective": objective_value,
                    "solve_seconds": solve_seconds,
                }
            )
    finally:
        if temp_config.exists():
            temp_config.unlink()

    summary_path = results_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(results_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n===== EXPERIMENT SUMMARY =====")
    print(
        f"{'Instance':<35} {'Scale':<8} {'Cust':>5} {'Pairs':>5} "
        f"{'Status':<12} {'Time(s)':>8} {'Obj':>14}"
    )
    print("-" * 102)
    for row in results_summary:
        objective_str = (
            f"{float(row['optimized_objective']):.4f}"
            if row.get("optimized_objective") is not None
            else "NA"
        )
        print(
            f"{str(row['instance']):<35} {str(row['scale']):<8} "
            f"{str(row['customers']):>5} {str(row['vehicle_pairs']):>5} "
            f"{str(row['status']):<12} {float(row['elapsed_seconds']):>8.1f} {objective_str:>14}"
        )

    ok_count = sum(1 for row in results_summary if row["status"] == "OK")
    print(f"\nTotal: {ok_count}/{total} succeeded")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    run_all_experiments()
