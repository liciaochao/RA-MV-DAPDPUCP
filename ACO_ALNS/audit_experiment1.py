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

ROOT = Path(r"D:\experiment\experiment_1")
REPORT_PATH = Path(r"D:\experiment\experiment_1_audit.md")
LARGE_FILE_BYTES = 10 * 1024 * 1024


def escape_md(text):
    return str(text).replace("|", "\\|")


def cell_to_text(value, max_len=120):
    if pd.isna(value):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "...(truncated)"
    return text


def df_to_markdown_table(df, max_rows=3):
    if df is None or df.empty:
        return "(empty)"

    preview = df.head(max_rows).copy()
    cols = [str(c) for c in preview.columns]

    lines = []
    lines.append("| " + " | ".join(escape_md(c) for c in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for _, row in preview.iterrows():
        vals = [escape_md(cell_to_text(v)) for v in row.tolist()]
        lines.append("| " + " | ".join(vals) + " |")

    return "\n".join(lines)


def read_text_with_fallback(path):
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    errors = []
    for enc in encodings:
        try:
            return path.read_text(encoding=enc), enc, None
        except Exception as e:
            errors.append(f"{enc}: {e}")
    return None, None, "; ".join(errors)


def list_json_like_dict(data_dict):
    if "records" in data_dict and isinstance(data_dict["records"], list):
        return "records", data_dict["records"]
    if "customers" in data_dict and isinstance(data_dict["customers"], list):
        return "customers", data_dict["customers"]

    for k, v in data_dict.items():
        if isinstance(v, list):
            if len(v) == 0:
                return k, v
            if isinstance(v[0], dict):
                return k, v
    return None, None


def json_to_dataframe(data):
    if isinstance(data, list):
        if len(data) == 0:
            return pd.DataFrame(), 0, "json_list_empty"
        if isinstance(data[0], dict):
            return pd.DataFrame(data), len(data), "json_list_of_dict"
        return pd.DataFrame({"value": data}), len(data), "json_list_scalar"

    if isinstance(data, dict):
        list_key, list_val = list_json_like_dict(data)
        if list_key is not None:
            if len(list_val) == 0:
                return pd.DataFrame(), 0, f"json_dict_list:{list_key}"
            if isinstance(list_val[0], dict):
                return pd.DataFrame(list_val), len(list_val), f"json_dict_list:{list_key}"
            return pd.DataFrame({list_key: list_val}), len(list_val), f"json_dict_list:{list_key}"

        normalized = {}
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                try:
                    normalized[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    normalized[k] = str(v)
            else:
                normalized[k] = v
        return pd.DataFrame([normalized]), 1, "json_dict_single_row"

    return pd.DataFrame([{"value": data}]), 1, "json_scalar"


def infer_purpose(path_obj, columns):
    fname = path_obj.name.lower()
    rel = str(path_obj).lower()
    col_l = [str(c).lower() for c in columns]

    if "comparison_summary" in fname:
        return "多算法对比汇总（含目标值/Gap/时间等）"
    if "experiment_summary" in fname:
        return "实验层汇总结果（多算例的总体统计）"
    if "gurobi_result" in fname or "gurobi" in rel:
        return "MILP/Gurobi 单算例求解结果"
    if "convergence" in fname or any("iter" in c or "iteration" in c for c in col_l):
        return "收敛过程记录（迭代过程数据）"
    if "objective" in fname or any("objective" in c or "z_total" in c for c in col_l):
        return "目标函数结果摘要"
    if "timing" in fname or any("time" in c or "second" in c for c in col_l):
        return "求解时间统计/分解"
    if "cost_breakdown" in fname or any("cost" in c for c in col_l):
        return "成本构成分解"
    if "solution_detail" in fname:
        return "解结构详情（路线/车辆对/服务信息）"
    if "customer_service" in fname:
        return "客户服务情况明细"
    if "mc_results" in fname:
        return "蒙特卡洛结果统计"
    if "online_execution_report" in fname:
        return "在线执行结果统计"
    return "用途待人工确认"


def has_gap_signal(columns, path_obj):
    name = path_obj.name.lower()
    if "gap" in name:
        return True
    for c in columns:
        if "gap" in str(c).lower():
            return True
    return False


def has_convergence_signal(columns, path_obj):
    name = path_obj.name.lower()
    if "convergence" in name:
        return True
    for c in columns:
        c_l = str(c).lower()
        if "iter" in c_l or "iteration" in c_l:
            return True
    return False


def has_timing_signal(columns, path_obj):
    name = path_obj.name.lower()
    if "timing" in name or "time" in name:
        return True
    for c in columns:
        c_l = str(c).lower()
        if "time" in c_l or "second" in c_l:
            return True
    return False


def is_instance_dir_name(name):
    parts = name.split("_")
    if len(parts) < 4:
        return False
    if parts[0] not in ["small", "medium", "large"]:
        return False
    if not parts[1].endswith("c"):
        return False
    if not parts[2].endswith("v"):
        return False
    if not parts[3].startswith("seed"):
        return False

    try:
        int(parts[1][:-1])
        int(parts[2][:-1])
        int(parts[3].replace("seed", ""))
        return True
    except Exception:
        return False


def is_per_instance_file(rel_parts, file_name):
    fn = file_name.lower()
    if "_seed" in fn:
        return True
    for p in rel_parts:
        if "_seed" in p.lower():
            return True
    return False


def build_tree_lines(root_path):
    lines = [root_path.name]

    def walk(current, prefix=""):
        try:
            entries = sorted(list(current.iterdir()), key=lambda x: (x.is_file(), x.name.lower()))
        except Exception:
            return

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            lines.append(prefix + connector + entry.name)
            if entry.is_dir():
                child_prefix = prefix + ("    " if is_last else "│   ")
                walk(entry, child_prefix)

    walk(root_path)
    return lines


def summarize_nulls(null_counts):
    if not null_counts:
        return "无"
    pairs = []
    for k, v in null_counts.items():
        pairs.append(f"{k}:{v}")
    return "; ".join(pairs)


def main():
    if not ROOT.exists() or not ROOT.is_dir():
        raise FileNotFoundError(f"Directory not found: {ROOT}")

    all_files = sorted([p for p in ROOT.rglob("*") if p.is_file()], key=lambda x: str(x).lower())
    all_dirs = sorted([p for p in ROOT.rglob("*") if p.is_dir()], key=lambda x: str(x).lower())

    ext_counts = {}
    for f in all_files:
        ext = f.suffix.lower() if f.suffix else "(no_ext)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    tracked_types = [".csv", ".json", ".png", ".txt"]

    top_level_dirs = [p for p in ROOT.iterdir() if p.is_dir()]
    top_level_dir_names = [p.name for p in sorted(top_level_dirs, key=lambda x: x.name.lower())]

    instance_dirs = []
    non_pattern_dirs = []
    scale_counts = {}
    algo_instance_counts = {}

    for d in all_dirs:
        rel_parts = d.relative_to(ROOT).parts
        if len(rel_parts) == 0:
            continue
        name = d.name
        if is_instance_dir_name(name):
            instance_dirs.append(d)
            scale = name.split("_")[0]
            scale_counts[scale] = scale_counts.get(scale, 0) + 1

            top_algo = rel_parts[0] if len(rel_parts) > 0 else "(unknown)"
            algo_instance_counts[top_algo] = algo_instance_counts.get(top_algo, 0) + 1
        else:
            if len(rel_parts) >= 2:
                non_pattern_dirs.append(d)

    data_files = [p for p in all_files if p.suffix.lower() in [".csv", ".json"]]

    records = []
    parse_errors = []

    structure_groups = {}
    per_instance_files = []
    summary_files = []
    convergence_files = []
    gap_files = []
    timing_files = []
    gurobi_related_files = []

    for idx, p in enumerate(sorted(data_files, key=lambda x: str(x).lower()), start=1):
        rel_path = str(p.relative_to(ROOT)).replace("\\", "/")
        file_size = p.stat().st_size
        is_large = file_size > LARGE_FILE_BYTES

        record = {
            "rel_path": rel_path,
            "ext": p.suffix.lower(),
            "rows": None,
            "columns": [],
            "preview": "",
            "has_null": False,
            "null_counts": {},
            "null_note": "",
            "purpose": "",
            "parse_error": None,
            "parse_mode": "",
            "large_limited": is_large,
        }

        try:
            if p.suffix.lower() == ".csv":
                if is_large:
                    df = pd.read_csv(p, nrows=100)
                    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                        line_count = sum(1 for _ in fh)
                    row_count = max(line_count - 1, 0)
                    null_df = df
                    record["null_note"] = "文件>10MB，空值统计基于前100行"
                else:
                    df = pd.read_csv(p)
                    row_count = int(df.shape[0])
                    null_df = df

                record["rows"] = row_count
                record["columns"] = [str(c) for c in df.columns.tolist()]

                null_series = null_df.isna().sum()
                null_series = null_series[null_series > 0]
                record["null_counts"] = {str(k): int(v) for k, v in null_series.to_dict().items()}
                record["has_null"] = len(record["null_counts"]) > 0
                record["preview"] = df_to_markdown_table(df.head(3), max_rows=3)
                record["parse_mode"] = "csv_full" if not is_large else "csv_large_preview100"

            else:
                text, used_enc, text_err = read_text_with_fallback(p)
                if text is None:
                    raise ValueError(f"读取文本失败: {text_err}")

                try:
                    data = json.loads(text)
                except Exception as e:
                    raise ValueError(f"JSON解析失败: {e}")

                df, row_count, parse_mode = json_to_dataframe(data)

                if is_large and len(df) > 100:
                    null_df = df.head(100)
                    preview_df = df.head(3)
                    record["null_note"] = "文件>10MB，空值统计基于前100行"
                else:
                    null_df = df
                    preview_df = df.head(3)

                record["rows"] = int(row_count)
                record["columns"] = [str(c) for c in df.columns.tolist()]

                if not null_df.empty:
                    null_series = null_df.isna().sum()
                    null_series = null_series[null_series > 0]
                    record["null_counts"] = {str(k): int(v) for k, v in null_series.to_dict().items()}
                    record["has_null"] = len(record["null_counts"]) > 0
                else:
                    record["null_counts"] = {}
                    record["has_null"] = False

                record["preview"] = df_to_markdown_table(preview_df, max_rows=3)
                record["parse_mode"] = f"{parse_mode};encoding={used_enc}"

            record["purpose"] = infer_purpose(p, record["columns"])

        except Exception as e:
            record["parse_error"] = str(e)
            record["purpose"] = "解析失败，需人工确认"
            parse_errors.append({"path": rel_path, "error": str(e)})

        records.append(record)

        if record["parse_error"] is None:
            signature = (record["ext"], tuple(record["columns"]))
            structure_groups.setdefault(signature, []).append(record["rel_path"])

            rel_parts = Path(record["rel_path"]).parts
            if is_per_instance_file(rel_parts, Path(record["rel_path"]).name):
                per_instance_files.append(record["rel_path"])
            else:
                summary_files.append(record["rel_path"])

            if has_convergence_signal(record["columns"], p):
                convergence_files.append(record["rel_path"])
            if has_gap_signal(record["columns"], p):
                gap_files.append(record["rel_path"])
            if has_timing_signal(record["columns"], p):
                timing_files.append(record["rel_path"])
            if "gurobi" in record["rel_path"].lower() or "gurobi" in p.name.lower():
                gurobi_related_files.append(record["rel_path"])

    same_structure_groups = []
    for (ext, cols), paths in structure_groups.items():
        if len(paths) > 1:
            same_structure_groups.append({
                "ext": ext,
                "columns": cols,
                "count": len(paths),
                "sample_paths": paths[:5],
            })
    same_structure_groups = sorted(same_structure_groups, key=lambda x: x["count"], reverse=True)

    tree_lines = build_tree_lines(ROOT)

    report = []
    report.append("# experiment_1 数据侦察报告")
    report.append("")

    report.append("## 1. 文件树")
    report.append("")
    report.append("```text")
    report.extend(tree_lines)
    report.append("```")
    report.append("")

    report.append("## 2. 文件类型统计")
    report.append("")
    report.append("| 文件类型 | 数量 |")
    report.append("|---|---:|")
    for ext in tracked_types:
        report.append(f"| {ext} | {ext_counts.get(ext, 0)} |")

    other_exts = [k for k in sorted(ext_counts.keys()) if k not in tracked_types]
    for ext in other_exts:
        report.append(f"| {ext} | {ext_counts.get(ext, 0)} |")

    report.append("")
    report.append(f"总文件数：`{len(all_files)}`")
    report.append("")

    report.append("## 3. 子文件夹命名规律")
    report.append("")
    report.append(f"顶层算法目录：`{', '.join(top_level_dir_names)}`")
    report.append("")
    report.append("实例目录命名主模式推断：`<scale>_<Nc>_<Nv>_seed<S>`")
    report.append(f"- 匹配该模式的实例目录数量：`{len(instance_dirs)}`")
    report.append(f"- 不匹配但位于更深层的目录数量：`{len(non_pattern_dirs)}`")
    report.append("")
    report.append("按规模标签统计：")
    for scale in ["small", "medium", "large"]:
        report.append(f"- {scale}: `{scale_counts.get(scale, 0)}`")
    report.append("")
    report.append("按顶层算法目录统计匹配实例目录数量：")
    for algo in sorted(algo_instance_counts.keys()):
        report.append(f"- {algo}: `{algo_instance_counts[algo]}`")
    report.append("")

    report.append("## 4. 各CSV/JSON文件逐一说明")
    report.append("")
    report.append("| 文件路径 | 列名 | 行数 | 推断用途 | 是否含空值 |")
    report.append("|---|---|---:|---|---|")

    for r in records:
        cols_text = ", ".join(r["columns"]) if r["columns"] else "(none)"
        if len(cols_text) > 180:
            cols_text = cols_text[:180] + "...(truncated)"
        rows_text = "ERR" if r["rows"] is None else str(r["rows"])
        null_text = "ERR" if r["parse_error"] else ("是" if r["has_null"] else "否")
        purpose_text = r["purpose"]
        if r["parse_error"]:
            purpose_text = f"解析失败: {r['parse_error']}"

        report.append(
            f"| {escape_md(r['rel_path'])} | {escape_md(cols_text)} | {rows_text} | {escape_md(purpose_text)} | {null_text} |"
        )

    report.append("")
    report.append("### 4.1 每个CSV/JSON文件前3行预览")
    report.append("")

    for r in records:
        report.append(f"#### `{r['rel_path']}`")
        if r["parse_error"]:
            report.append(f"- 解析状态：失败")
            report.append(f"- 报错：`{escape_md(r['parse_error'])}`")
            report.append("")
            continue

        report.append(f"- 解析模式：`{escape_md(r['parse_mode'])}`")
        report.append(f"- 行数：`{r['rows']}`")
        report.append(f"- 列名：`{escape_md(', '.join(r['columns']) if r['columns'] else '(none)')}`")
        report.append(f"- 是否含空值：`{'是' if r['has_null'] else '否'}`")
        report.append(f"- 缺失字段统计：`{escape_md(summarize_nulls(r['null_counts']))}`")
        if r["null_note"]:
            report.append(f"- 备注：`{escape_md(r['null_note'])}`")
        report.append(f"- 推断用途：`{escape_md(r['purpose'])}`")
        report.append("")
        report.append(r["preview"])
        report.append("")

    report.append("## 5. 关键发现")
    report.append("")

    report.append("结构相同（可合并读取）的主要文件组（按组内文件数降序，最多显示前20组）：")
    report.append("")
    if same_structure_groups:
        report.append("| 扩展名 | 组内文件数 | 列名数量 | 示例文件 |")
        report.append("|---|---:|---:|---|")
        for g in same_structure_groups[:20]:
            col_count = len(g["columns"])
            example = "; ".join(g["sample_paths"][:3])
            report.append(f"| {g['ext']} | {g['count']} | {col_count} | {escape_md(example)} |")
    else:
        report.append("- 未发现可分组的同结构文件")

    report.append("")

    unique_per_instance_names = sorted({Path(p).name for p in per_instance_files})
    unique_summary_names = sorted({Path(p).name for p in summary_files})

    report.append(f"- 每算例一份文件（按文件名去重）数量：`{len(unique_per_instance_names)}`")
    report.append(f"- 汇总型文件（按文件名去重）数量：`{len(unique_summary_names)}`")
    report.append("")

    report.append("每算例一份（去重文件名）：")
    if unique_per_instance_names:
        for n in unique_per_instance_names:
            report.append(f"- {n}")
    else:
        report.append("- (none)")
    report.append("")

    report.append("汇总型（去重文件名）：")
    if unique_summary_names:
        for n in unique_summary_names:
            report.append(f"- {n}")
    else:
        report.append("- (none)")
    report.append("")

    report.append("可用于Gap对比分析的文件：")
    if gap_files:
        for p in sorted(gap_files):
            report.append(f"- {p}")
    else:
        report.append("- 未发现")
    report.append("")

    report.append("可用于收敛曲线的文件：")
    if convergence_files:
        for p in sorted(convergence_files):
            report.append(f"- {p}")
    else:
        report.append("- 未发现")
    report.append("")

    report.append("可用于求解时间对比的文件：")
    if timing_files:
        for p in sorted(timing_files):
            report.append(f"- {p}")
    else:
        report.append("- 未发现")
    report.append("")

    report.append("存在疑问/需人工确认的数据：")
    unknown_files = [r["rel_path"] for r in records if r["purpose"] == "用途待人工确认"]
    if parse_errors:
        for e in parse_errors:
            report.append(f"- 解析失败：{e['path']} -> {e['error']}")
    if unknown_files:
        for p in unknown_files:
            report.append(f"- 用途未自动识别：{p}")
    if (not parse_errors) and (not unknown_files):
        report.append("- 无")
    report.append("")

    report.append("是否存在记录收敛过程（含迭代列）文件：")
    report.append(f"- {'是' if len(convergence_files) > 0 else '否'}（数量：{len(convergence_files)}）")
    report.append("")

    report.append("是否存在记录各算法对比结果文件：")
    comparison_hits = [r["rel_path"] for r in records if "comparison" in Path(r["rel_path"]).name.lower()]
    if comparison_hits:
        report.append(f"- 是（数量：{len(comparison_hits)}）")
        for p in comparison_hits:
            report.append(f"- {p}")
    else:
        report.append("- 否")
    report.append("")

    report.append("是否存在与MILP/Gurobi结果相关的文件：")
    if gurobi_related_files:
        report.append(f"- 是（数量：{len(gurobi_related_files)}）")
        for p in sorted(gurobi_related_files):
            report.append(f"- {p}")
    else:
        report.append("- 否")
    report.append("")

    report.append("## 6. 不确定项清单（需原作者确认）")
    report.append("")

    uncertainties = []
    uncertainties.append("`objective_summary.json` 中 `note` 字段是否统一定义为与最优解Gap，或仅为文本说明。")
    uncertainties.append("`convergence_data.json` 的迭代字段单位（迭代轮次/时间步）及目标值字段是否可跨算法直接比较。")
    uncertainties.append("`timing.json` 与 `timing_report.json` 是否语义完全等价，是否存在一个为冗余导出。")
    uncertainties.append("`mc_results.json` 在实验一（确定性）中出现的定位：仅调试留存还是正式指标之一。")

    if parse_errors:
        for e in parse_errors:
            uncertainties.append(f"文件解析失败需确认编码/格式：{e['path']} -> {e['error']}")

    for u in uncertainties:
        report.append(f"- {u}")

    report.append("")
    report.append("---")
    report.append("")
    report.append(f"审查时间根目录：`{ROOT}`")
    report.append(f"共审查 CSV/JSON 文件数：`{len(records)}`")

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    csv_count = sum(1 for p in data_files if p.suffix.lower() == ".csv")
    print(f"CSV files found: {csv_count}")
    print(f"Convergence data found: {'Yes' if len(convergence_files) > 0 else 'No'}")
    print(f"Gap data found: {'Yes' if len(gap_files) > 0 else 'No'}")
    print(f"Report generated: {REPORT_PATH}")


if __name__ == "__main__":
    main()

