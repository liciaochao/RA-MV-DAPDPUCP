from pathlib import Path
import os
import json
import sys

try:
    import pandas as pd
except ModuleNotFoundError:
    vendor_path = Path(__file__).resolve().parent / ".vendor"
    if vendor_path.exists():
        sys.path.insert(0, str(vendor_path))
    import pandas as pd

BASE = Path(r"D:\experiment\experiment_1")
OUT_DIR = BASE / "output"

ALGO_MAP = {
    "Pure_ACO": "纯ACO",
    "Pure_ALNS": "纯ALNS",
    "ACO_ALNS": "ACO_ALNS",
}

SIZE_ORDER = {"Small": 0, "Medium": 1, "Large": 2}


class WarningCollector:
    def __init__(self):
        self.lines = []

    def add(self, msg):
        self.lines.append(msg)

    def save(self, path):
        text = "\n".join(self.lines)
        path.write_text(text, encoding="utf-8")


def read_json(path, warnings):
    if not path.exists():
        warnings.add(f"MISSING|{path}")
        return None

    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            txt = path.read_text(encoding=enc)
            return json.loads(txt)
        except Exception as e:
            last_err = e
    warnings.add(f"PARSE_FAIL|{path}|{last_err}")
    return None


def parse_instance_name(name):
    parts = name.split("_")
    if len(parts) != 4:
        return None

    size_raw, nc_raw, nv_raw, seed_raw = parts

    if size_raw == "small":
        size = "Small"
    elif size_raw == "medium":
        size = "Medium"
    elif size_raw == "large":
        size = "Large"
    else:
        return None

    if (not nc_raw.endswith("c")) or (not nv_raw.endswith("v")) or (not seed_raw.startswith("seed")):
        return None

    try:
        n_customers = int(nc_raw[:-1])
        n_vehicles = int(nv_raw[:-1])
        seed = int(seed_raw.replace("seed", ""))
    except Exception:
        return None

    return {
        "instance": name,
        "size": size,
        "n_customers": n_customers,
        "n_vehicles": n_vehicles,
        "seed": seed,
    }


def describe_json(obj, indent=0, max_list_preview=3):
    sp = " " * indent
    lines = []

    if isinstance(obj, dict):
        lines.append(f"{sp}dict(keys={list(obj.keys())})")
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{sp}- {k}: {type(v).__name__}")
                lines.extend(describe_json(v, indent + 2, max_list_preview=max_list_preview))
            else:
                lines.append(f"{sp}- {k}: {type(v).__name__} = {v}")

    elif isinstance(obj, list):
        lines.append(f"{sp}list(len={len(obj)})")
        preview_n = min(len(obj), max_list_preview)
        for i in range(preview_n):
            item = obj[i]
            lines.append(f"{sp}- [{i}] {type(item).__name__}")
            if isinstance(item, (dict, list)):
                lines.extend(describe_json(item, indent + 2, max_list_preview=max_list_preview))
            else:
                lines.append(f"{sp}  value = {item}")
        if len(obj) > preview_n:
            lines.append(f"{sp}- ... {len(obj)-preview_n} more items")

    else:
        lines.append(f"{sp}{type(obj).__name__} = {obj}")

    return lines


def collect_numeric_fields_from_dict(d):
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def select_key_by_priority(columns, must_contains=None, priority_tokens=None, exclude_tokens=None):
    if must_contains is None:
        must_contains = []
    if priority_tokens is None:
        priority_tokens = []
    if exclude_tokens is None:
        exclude_tokens = []

    cols = [c for c in columns]
    cols_lower = {c: c.lower() for c in cols}

    candidates = []
    for c in cols:
        c_l = cols_lower[c]
        ok = True
        for t in must_contains:
            if t not in c_l:
                ok = False
                break
        if not ok:
            continue
        ex = False
        for t in exclude_tokens:
            if t in c_l:
                ex = True
                break
        if ex:
            continue
        candidates.append(c)

    if not candidates:
        return None

    def score(col):
        c_l = col.lower()
        s = 0
        for i, t in enumerate(priority_tokens):
            if t in c_l:
                s += (len(priority_tokens) - i) * 10
        s -= len(c_l) * 0.001
        return s

    candidates_sorted = sorted(candidates, key=score, reverse=True)
    return candidates_sorted[0]


def extract_objective_fields(obj_data):
    result = {
        "final_val": float("nan"),
        "final_key": None,
        "initial_val": float("nan"),
        "initial_key": None,
        "cost_parts": {},
    }

    if not isinstance(obj_data, dict):
        return result

    num_map = collect_numeric_fields_from_dict(obj_data)
    keys = list(num_map.keys())

    # final objective
    final_key = select_key_by_priority(
        keys,
        must_contains=[],
        priority_tokens=["optimized", "final", "best", "objective", "total", "cost"],
        exclude_tokens=["initial", "improvement", "percent", "gap", "delta"],
    )
    if final_key is None:
        fallback = []
        for k in keys:
            k_l = k.lower()
            if ("objective" in k_l) or ("cost" in k_l) or ("total" in k_l) or ("best" in k_l):
                fallback.append(k)
        if fallback:
            final_key = fallback[0]
    if final_key is None and len(keys) > 0:
        final_key = keys[0]

    if final_key is not None:
        result["final_key"] = final_key
        result["final_val"] = num_map[final_key]

    # initial objective
    initial_candidates = []
    for k in keys:
        k_l = k.lower()
        if "initial" in k_l and (("objective" in k_l) or ("cost" in k_l) or ("total" in k_l)):
            initial_candidates.append(k)
    if not initial_candidates:
        for k in keys:
            if "initial" in k.lower():
                initial_candidates.append(k)
    if initial_candidates:
        initial_key = initial_candidates[0]
        result["initial_key"] = initial_key
        result["initial_val"] = num_map[initial_key]

    # cost parts (if exist)
    part_tokens = ["z_fixed", "z_truck", "z_drone", "z_fail", "z_tw", "fixed", "truck", "drone", "fail", "penalty"]
    for k, v in num_map.items():
        k_l = k.lower()
        for t in part_tokens:
            if t in k_l:
                result["cost_parts"][k] = v
                break

    return result


def extract_timing_fields(timing_data):
    result = {
        "total_time": float("nan"),
        "total_key": None,
        "phase_times": {},
    }

    if not isinstance(timing_data, dict):
        return result

    num_map = collect_numeric_fields_from_dict(timing_data)
    keys = list(num_map.keys())

    total_key = select_key_by_priority(
        keys,
        priority_tokens=["total", "solve", "elapsed", "seconds", "time"],
    )
    if total_key is None:
        if len(keys) > 0:
            total_key = keys[0]

    if total_key is not None:
        result["total_key"] = total_key
        result["total_time"] = num_map[total_key]

    for k, v in num_map.items():
        k_l = k.lower()
        if any(t in k_l for t in ["initial", "opt", "cluster", "monte", "phase", "construct"]):
            result["phase_times"][k] = v

    return result


def safe_to_float(v):
    try:
        if pd.isna(v):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def extract_convergence_series(conv_data):
    result = {
        "iter_key": None,
        "obj_key": None,
        "rows": [],
        "pheromone_keys": [],
    }

    records = None

    if isinstance(conv_data, dict):
        if "records" in conv_data and isinstance(conv_data["records"], list):
            records = conv_data["records"]
        else:
            for _, v in conv_data.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                    records = v
                    break
    elif isinstance(conv_data, list):
        if len(conv_data) > 0 and isinstance(conv_data[0], dict):
            records = conv_data

    if records is None:
        return result

    if len(records) == 0:
        return result

    first = records[0]
    keys = list(first.keys())

    iter_key = select_key_by_priority(
        keys,
        priority_tokens=["iteration", "global_step", "iter", "generation", "step", "aco_iter"],
    )
    if iter_key is None:
        iter_key = keys[0]

    obj_key = select_key_by_priority(
        keys,
        priority_tokens=["best_objective", "best_cost", "objective", "current_cost", "cost", "obj", "value"],
    )
    if obj_key is None:
        obj_key = keys[0]

    pheromone_keys = []
    for k in keys:
        k_l = k.lower()
        if ("pheromone" in k_l) or ("tau" in k_l):
            pheromone_keys.append(k)

    rows = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        iv = rec.get(iter_key)
        ov = rec.get(obj_key)
        ivf = safe_to_float(iv)
        ovf = safe_to_float(ov)
        if pd.isna(ivf) or pd.isna(ovf):
            continue
        rows.append((int(ivf), float(ovf)))

    result["iter_key"] = iter_key
    result["obj_key"] = obj_key
    result["rows"] = rows
    result["pheromone_keys"] = pheromone_keys

    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings = WarningCollector()

    # ---------------- Step 0 ----------------
    step0_paths = [
        BASE / "纯ACO" / "small_8c_1v_seed1" / "objective_summary.json",
        BASE / "纯ACO" / "small_8c_1v_seed1" / "timing.json",
        BASE / "纯ACO" / "small_8c_1v_seed1" / "convergence_data.json",
        BASE / "纯ALNS" / "small_8c_1v_seed1" / "objective_summary.json",
        BASE / "纯ALNS" / "small_8c_1v_seed1" / "timing.json",
        BASE / "纯ALNS" / "small_8c_1v_seed1" / "convergence_data.json",
        BASE / "ACO_ALNS" / "small_8c_1v_seed1" / "objective_summary.json",
        BASE / "ACO_ALNS" / "small_8c_1v_seed1" / "timing.json",
        BASE / "ACO_ALNS" / "small_8c_1v_seed1" / "convergence_data.json",
    ]

    print("===== Step 0: Field Structure Inspection =====")
    for p in step0_paths:
        print(f"\n--- FILE: {p} ---")
        obj = read_json(p, warnings)
        if obj is None:
            print("[ERROR] Cannot parse JSON")
            continue
        lines = describe_json(obj, indent=0, max_list_preview=3)
        for ln in lines:
            print(ln)

    print("[Step 0 Done] 请人工确认字段名后再继续 Step 1。")

    # ---------------- Step 1 ----------------
    algo_dirs = {algo: BASE / folder for algo, folder in ALGO_MAP.items()}
    dir_sets = {}

    for algo, pdir in algo_dirs.items():
        names = []
        if pdir.exists() and pdir.is_dir():
            for c in pdir.iterdir():
                if c.is_dir():
                    names.append(c.name)
        else:
            warnings.add(f"MISSING_ALGO_DIR|{pdir}")
        dir_sets[algo] = set(names)

    union_instances = set()
    for s in dir_sets.values():
        union_instances |= s

    # consistency check
    base_algo = "ACO_ALNS"
    base_set = dir_sets.get(base_algo, set())
    for algo, s in dir_sets.items():
        only_in_base = sorted(list(base_set - s))
        only_in_algo = sorted(list(s - base_set))
        if only_in_base or only_in_algo:
            warnings.add(f"INSTANCE_SET_MISMATCH|baseline={base_algo}|algo={algo}|missing_in_algo={only_in_base}|extra_in_algo={only_in_algo}")

    parsed_instances = []
    bad_names = []
    for inst in sorted(union_instances):
        parsed = parse_instance_name(inst)
        if parsed is None:
            bad_names.append(inst)
            warnings.add(f"BAD_INSTANCE_NAME|{inst}")
            continue
        parsed_instances.append(parsed)

    parsed_instances = sorted(
        parsed_instances,
        key=lambda r: (SIZE_ORDER.get(r["size"], 99), r["n_customers"], r["seed"], r["instance"]),
    )

    instances_list_path = OUT_DIR / "instances_list.txt"
    instances_list_path.write_text("\n".join([r["instance"] for r in parsed_instances]), encoding="utf-8")
    print(f"[Step 1 Done] instances_list.txt saved. ({len(parsed_instances)}/{len(union_instances)})")

    # ---------------- Step 2 ----------------
    objective_success = 0
    timing_success = 0
    convergence_success = 0

    total_expected = len(parsed_instances) * len(ALGO_MAP)

    field_mapping_records = []

    main_rows = []
    conv_rows = []

    for inst_meta in parsed_instances:
        inst = inst_meta["instance"]

        row = {
            "instance": inst,
            "size": inst_meta["size"],
            "n_customers": inst_meta["n_customers"],
            "n_vehicles": inst_meta["n_vehicles"],
            "seed": inst_meta["seed"],
            "ACO_obj_final": float("nan"),
            "ALNS_obj_final": float("nan"),
            "ACO_ALNS_obj_final": float("nan"),
            "delta_vs_ACO_pct": float("nan"),
            "delta_vs_ALNS_pct": float("nan"),
            "ACO_time_s": float("nan"),
            "ALNS_time_s": float("nan"),
            "ACO_ALNS_time_s": float("nan"),
            "ACO_obj_initial": float("nan"),
            "ALNS_obj_initial": float("nan"),
            "ACO_ALNS_obj_initial": float("nan"),
        }

        for algo_label, folder_name in ALGO_MAP.items():
            folder = BASE / folder_name / inst

            obj_file = folder / "objective_summary.json"
            time_file = folder / "timing.json"
            conv_file = folder / "convergence_data.json"

            obj_data = read_json(obj_file, warnings)
            time_data = read_json(time_file, warnings)
            conv_data = read_json(conv_file, warnings)

            obj_info = extract_objective_fields(obj_data) if obj_data is not None else None
            time_info = extract_timing_fields(time_data) if time_data is not None else None
            conv_info = extract_convergence_series(conv_data) if conv_data is not None else None

            if obj_info is not None:
                objective_success += 1
                field_mapping_records.append({
                    "file_type": "objective_summary",
                    "algorithm": algo_label,
                    "selected_final_key": obj_info["final_key"],
                    "selected_initial_key": obj_info["initial_key"],
                })

                if algo_label == "Pure_ACO":
                    row["ACO_obj_final"] = obj_info["final_val"]
                    row["ACO_obj_initial"] = obj_info["initial_val"]
                elif algo_label == "Pure_ALNS":
                    row["ALNS_obj_final"] = obj_info["final_val"]
                    row["ALNS_obj_initial"] = obj_info["initial_val"]
                elif algo_label == "ACO_ALNS":
                    row["ACO_ALNS_obj_final"] = obj_info["final_val"]
                    row["ACO_ALNS_obj_initial"] = obj_info["initial_val"]

            if time_info is not None:
                timing_success += 1
                field_mapping_records.append({
                    "file_type": "timing",
                    "algorithm": algo_label,
                    "selected_total_key": time_info["total_key"],
                })
                if algo_label == "Pure_ACO":
                    row["ACO_time_s"] = time_info["total_time"]
                elif algo_label == "Pure_ALNS":
                    row["ALNS_time_s"] = time_info["total_time"]
                elif algo_label == "ACO_ALNS":
                    row["ACO_ALNS_time_s"] = time_info["total_time"]

            if conv_info is not None:
                convergence_success += 1
                field_mapping_records.append({
                    "file_type": "convergence",
                    "algorithm": algo_label,
                    "selected_iter_key": conv_info["iter_key"],
                    "selected_obj_key": conv_info["obj_key"],
                    "pheromone_keys": ",".join(conv_info["pheromone_keys"]) if conv_info["pheromone_keys"] else "",
                })
                for iv, ov in conv_info["rows"]:
                    conv_rows.append({
                        "instance": inst,
                        "size": inst_meta["size"],
                        "n_customers": inst_meta["n_customers"],
                        "algorithm": algo_label,
                        "iteration": iv,
                        "obj_value": ov,
                    })

        # delta metrics
        aco = row["ACO_obj_final"]
        alns = row["ALNS_obj_final"]
        hybrid = row["ACO_ALNS_obj_final"]

        if (not pd.isna(aco)) and aco != 0 and (not pd.isna(hybrid)):
            row["delta_vs_ACO_pct"] = (aco - hybrid) / aco * 100.0
        if (not pd.isna(alns)) and alns != 0 and (not pd.isna(hybrid)):
            row["delta_vs_ALNS_pct"] = (alns - hybrid) / alns * 100.0

        main_rows.append(row)

    warnings_log_path = OUT_DIR / "warnings.log"
    warnings.save(warnings_log_path)
    print(f"[Step 2 Done] warnings.log saved. ({objective_success + timing_success + convergence_success}/{total_expected * 3})")

    # ---------------- Step 3 ----------------
    main_df = pd.DataFrame(main_rows)
    main_df = main_df.sort_values(by=["size", "n_customers", "seed"], key=lambda s: s.map(SIZE_ORDER) if s.name == "size" else s)

    main_cols = [
        "instance", "size", "n_customers", "n_vehicles", "seed",
        "ACO_obj_final", "ALNS_obj_final", "ACO_ALNS_obj_final",
        "delta_vs_ACO_pct", "delta_vs_ALNS_pct",
        "ACO_time_s", "ALNS_time_s", "ACO_ALNS_time_s",
        "ACO_obj_initial", "ALNS_obj_initial", "ACO_ALNS_obj_initial",
    ]
    main_df = main_df[main_cols]

    main_path = OUT_DIR / "exp1_main_table.csv"
    main_df.to_csv(main_path, index=False, encoding="utf-8-sig")

    conv_df = pd.DataFrame(conv_rows)
    conv_cols = ["instance", "size", "n_customers", "algorithm", "iteration", "obj_value"]
    if conv_df.empty:
        conv_df = pd.DataFrame(columns=conv_cols)
    else:
        conv_df = conv_df[conv_cols]
        conv_df = conv_df.sort_values(by=["size", "n_customers", "instance", "algorithm", "iteration"], key=lambda s: s.map(SIZE_ORDER) if s.name == "size" else s)

    conv_path = OUT_DIR / "exp1_convergence_combined.csv"
    conv_df.to_csv(conv_path, index=False, encoding="utf-8-sig")

    # grouped summary
    long_rows = []
    for _, r in main_df.iterrows():
        long_rows.append({
            "instance": r["instance"],
            "size": r["size"],
            "algorithm": "Pure_ACO",
            "obj": r["ACO_obj_final"],
            "time": r["ACO_time_s"],
            "delta": r["delta_vs_ACO_pct"],
        })
        long_rows.append({
            "instance": r["instance"],
            "size": r["size"],
            "algorithm": "Pure_ALNS",
            "obj": r["ALNS_obj_final"],
            "time": r["ALNS_time_s"],
            "delta": r["delta_vs_ALNS_pct"],
        })
        long_rows.append({
            "instance": r["instance"],
            "size": r["size"],
            "algorithm": "ACO_ALNS",
            "obj": r["ACO_ALNS_obj_final"],
            "time": r["ACO_ALNS_time_s"],
            "delta": float("nan"),
        })

    long_df = pd.DataFrame(long_rows)
    g = long_df.groupby(["size", "algorithm"], dropna=False)

    summary_df = g.agg(
        n_instances=("instance", "nunique"),
        obj_mean=("obj", "mean"),
        obj_min=("obj", "min"),
        obj_max=("obj", "max"),
        delta_mean=("delta", "mean"),
        delta_min=("delta", "min"),
        delta_max=("delta", "max"),
        time_mean=("time", "mean"),
        time_min=("time", "min"),
        time_max=("time", "max"),
    ).reset_index()

    summary_df = summary_df.sort_values(by=["size", "algorithm"], key=lambda s: s.map(SIZE_ORDER) if s.name == "size" else s)

    summary_path = OUT_DIR / "exp1_grouped_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"[Step 3 Done] exp1_main_table.csv, exp1_convergence_combined.csv, exp1_grouped_summary.csv saved. ({len(main_df)}/{len(parsed_instances)})")

    # ---------------- Step 4 ----------------
    # Completeness
    expected_files_per_type = len(parsed_instances) * len(ALGO_MAP)

    # Parse warnings for missing/parse fail counts by file type
    missing_or_failed = [ln for ln in warnings.lines if ln.startswith("MISSING|") or ln.startswith("PARSE_FAIL|")]

    obj_read_ok = objective_success
    time_read_ok = timing_success
    conv_read_ok = convergence_success

    # anomalies
    bad_obj = main_df[(main_df["ACO_obj_final"] <= 0) | (main_df["ALNS_obj_final"] <= 0) | (main_df["ACO_ALNS_obj_final"] <= 0)]

    bad_time = main_df[
        (main_df["ACO_time_s"] == 0) | (main_df["ALNS_time_s"] == 0) | (main_df["ACO_ALNS_time_s"] == 0) |
        (main_df["ACO_time_s"] > 3600) | (main_df["ALNS_time_s"] > 3600) | (main_df["ACO_ALNS_time_s"] > 3600)
    ]

    bad_delta = main_df[
        (main_df["delta_vs_ACO_pct"].abs() > 20) |
        (main_df["delta_vs_ALNS_pct"].abs() > 20)
    ]

    same_obj = main_df[
        (main_df["ACO_obj_final"] == main_df["ALNS_obj_final"]) &
        (main_df["ALNS_obj_final"] == main_df["ACO_ALNS_obj_final"])
    ]

    # field mapping summary
    fmap_df = pd.DataFrame(field_mapping_records)

    lines = []
    lines.append("# exp1 Data Quality Report")
    lines.append("")

    lines.append("## 1. 文件完整性检查")
    lines.append("")
    lines.append(f"- objective_summary.json: {obj_read_ok}/{expected_files_per_type}")
    lines.append(f"- timing.json: {time_read_ok}/{expected_files_per_type}")
    lines.append(f"- convergence_data.json: {conv_read_ok}/{expected_files_per_type}")
    lines.append("")
    lines.append("缺失或解析失败文件：")
    if missing_or_failed:
        for w in missing_or_failed:
            lines.append(f"- {w}")
    else:
        lines.append("- 无")

    lines.append("")
    lines.append("## 2. 数值异常检查")
    lines.append("")
    lines.append(f"- obj_final <= 0 记录数: {len(bad_obj)}")
    if len(bad_obj) > 0:
        for _, r in bad_obj.iterrows():
            lines.append(f"  - {r['instance']}")

    lines.append(f"- time_s == 0 或 > 3600 记录数: {len(bad_time)}")
    if len(bad_time) > 0:
        for _, r in bad_time.iterrows():
            lines.append(f"  - {r['instance']}")

    lines.append(f"- |delta| > 20% 记录数: {len(bad_delta)}")
    if len(bad_delta) > 0:
        for _, r in bad_delta.iterrows():
            lines.append(f"  - {r['instance']} | delta_vs_ACO={r['delta_vs_ACO_pct']} | delta_vs_ALNS={r['delta_vs_ALNS_pct']}")

    lines.append(f"- 三算法目标值完全相同 记录数: {len(same_obj)}")
    if len(same_obj) > 0:
        for _, r in same_obj.iterrows():
            lines.append(f"  - {r['instance']} | obj={r['ACO_obj_final']}")

    lines.append("")
    lines.append("## 3. 字段名称映射记录")
    lines.append("")
    if fmap_df.empty:
        lines.append("- 无可用映射记录")
    else:
        lines.append("### objective_summary.json")
        sub = fmap_df[fmap_df["file_type"] == "objective_summary"]
        if sub.empty:
            lines.append("- 无")
        else:
            t = sub.groupby(["algorithm", "selected_final_key", "selected_initial_key"]).size().reset_index(name="count")
            for _, r in t.iterrows():
                lines.append(f"- algo={r['algorithm']}, final={r['selected_final_key']}, initial={r['selected_initial_key']}, count={r['count']}")

        lines.append("### timing.json")
        sub = fmap_df[fmap_df["file_type"] == "timing"]
        if sub.empty:
            lines.append("- 无")
        else:
            t = sub.groupby(["algorithm", "selected_total_key"]).size().reset_index(name="count")
            for _, r in t.iterrows():
                lines.append(f"- algo={r['algorithm']}, total={r['selected_total_key']}, count={r['count']}")

        lines.append("### convergence_data.json")
        sub = fmap_df[fmap_df["file_type"] == "convergence"]
        if sub.empty:
            lines.append("- 无")
        else:
            t = sub.groupby(["algorithm", "selected_iter_key", "selected_obj_key", "pheromone_keys"]).size().reset_index(name="count")
            for _, r in t.iterrows():
                lines.append(
                    f"- algo={r['algorithm']}, iter={r['selected_iter_key']}, obj={r['selected_obj_key']}, pheromone_keys={r['pheromone_keys']}, count={r['count']}"
                )

    lines.append("")
    lines.append("## 4. 简要统计摘要")
    lines.append("")

    # mean objective by size and algorithm
    if not long_df.empty:
        tmp = long_df.groupby(["size", "algorithm"], dropna=False)["obj"].mean().reset_index()
        lines.append("### 三算法在Small/Medium/Large上的平均目标值")
        for _, r in tmp.sort_values(by=["size", "algorithm"], key=lambda s: s.map(SIZE_ORDER) if s.name == "size" else s).iterrows():
            lines.append(f"- size={r['size']}, algo={r['algorithm']}, obj_mean={r['obj']}")

    avg_delta_aco = main_df["delta_vs_ACO_pct"].mean()
    avg_delta_alns = main_df["delta_vs_ALNS_pct"].mean()
    lines.append("### ACO-ALNS相对两基准的平均改进率")
    lines.append(f"- vs Pure_ACO: {avg_delta_aco}")
    lines.append(f"- vs Pure_ALNS: {avg_delta_alns}")

    quality_path = OUT_DIR / "exp1_data_quality.md"
    quality_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[Step 4 Done] exp1_data_quality.md saved. ({len(parsed_instances)}/{len(parsed_instances)})")


if __name__ == "__main__":
    main()

