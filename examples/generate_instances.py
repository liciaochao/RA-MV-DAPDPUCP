#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
按用户给定规则生成两套算例（deterministic + stochastic），并完成校验与报告输出。
仅依赖 Python 标准库：json、random、copy、math、os（另含 re/shutil/sys/importlib/datetime，均为标准库）。
"""

import copy
import importlib
import json
import math
import os
import random
import re
import shutil
import sys
from datetime import datetime


# ====== 路径常量（按要求硬编码） ======
EXISTING_DIR = r"D:\codex_drone\examples\instances"
OUTPUT_ROOT = r"D:\examples"
DET_DIR = r"D:\examples\deterministic"
STO_DIR = r"D:\examples\stochastic"
REPORT_PATH = r"D:\examples\generation_report.txt"
SCRIPT_OUTPUT_PATH = r"D:\examples\generate_instances.py"
SOURCE_ROOT = r"D:\codex_drone\src"

# 手工排除需要“替换”为新规模的已有算例文件
EXCLUDE_EXISTING_FILENAMES = {
    "large_300c_18v_seed1.json",
    "large_300c_18v_seed2.json",
    "large_300c_18v_seed3.json",
}


# ====== 文件名解析 ======
FILENAME_RE = re.compile(r"^(small|medium|large)_(\d+)c_(\d+)v_seed(\d+)\.json$")


# ====== 模板字段定义（严格字段名/层次） ======
TOP_LEVEL_KEYS = [
    "depot_id",
    "vehicle_pair_count",
    "depot",
    "truck_distance_scale",
    "drone_distance_scale",
    "trucks",
    "drones",
    "delivery_customers",
    "pickup_customers",
    "customers",
]
DEPOT_KEYS = {"x", "y"}
TRUCK_KEYS = {"truck_id", "speed", "capacity"}
DRONE_KEYS = {
    "drone_id",
    "speed",
    "capacity",
    "max_range_km",
    "energy_wh",
    "launch_time",
    "recovery_time",
}
CUSTOMER_KEYS = {
    "customer_id",
    "x",
    "y",
    "customer_type",
    "weight",
    "time_window",
    "service_time",
    "home_probabilities",
}
HP_ONES = [1.0] * 8


# ====== 目标算例清单（按用户给定） ======
TARGET_SPECS = [
    # small
    (8, 1, [1, 2, 3]),
    (10, 1, [1, 2, 3]),
    (12, 1, [1, 2, 3]),
    (15, 1, [1, 2, 3]),
    # 为补足小规模算例数量，补充 16 客户组（seed1~seed3）
    (16, 1, [1, 2, 3]),
    (18, 2, [1, 2, 3]),
    (20, 2, [1, 2, 3]),
    # medium
    (30, 2, [1, 2, 3]),
    (40, 3, [1, 2, 3]),
    (50, 3, [1, 2, 3]),
    (60, 4, [1, 2, 3]),
    (80, 5, [1, 2, 3]),
    # large
    (100, 7, [1, 2, 3]),
    (150, 9, [1, 2, 3]),
    (180, 11, [1, 2, 3]),
    (200, 13, [1, 2, 3]),
    (250, 15, [1, 2, 3]),
    (280, 18, [1, 2, 3]),
    (300, 20, [1, 2, 3]),
]

SEED_DISTRIBUTION = {
    1: "clustered",
    2: "random",
    3: "mixed",
}


def ensure_output_dirs():
    """创建输出目录。"""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(DET_DIR, exist_ok=True)
    os.makedirs(STO_DIR, exist_ok=True)


def clear_json_files(target_dir):
    """清空目标目录下旧的 JSON 文件，避免残留影响一致性验证。"""
    for name in os.listdir(target_dir):
        if name.lower().endswith(".json"):
            path = os.path.join(target_dir, name)
            if os.path.isfile(path):
                os.remove(path)


def parse_filename(fname):
    """解析文件名中的规模/客户数/车辆对数/seed。"""
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    scale, n_str, v_str, s_str = m.groups()
    return {
        "scale": scale,
        "n_customers": int(n_str),
        "n_pairs": int(v_str),
        "seed": int(s_str),
        "filename": fname,
    }


def get_existing_inventory():
    """盘点已有算例。"""
    infos = []
    for name in sorted(os.listdir(EXISTING_DIR)):
        if not name.lower().endswith(".json"):
            continue
        if name in EXCLUDE_EXISTING_FILENAMES:
            continue
        parsed = parse_filename(name)
        if parsed is None:
            parsed = {
                "scale": "unknown",
                "n_customers": None,
                "n_pairs": None,
                "seed": None,
                "filename": name,
            }
        infos.append(parsed)
    return infos


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def validate_structure_against_template(data, fname):
    """检查单个实例是否满足模板字段结构（字段名和层次）。"""
    warnings = []

    top_set = set(data.keys()) if isinstance(data, dict) else set()
    expected_top = set(TOP_LEVEL_KEYS)
    missing_top = expected_top - top_set
    extra_top = top_set - expected_top
    if missing_top:
        warnings.append(f"{fname}: 缺少顶层字段 {sorted(missing_top)}")
    if extra_top:
        warnings.append(f"{fname}: 顶层存在额外字段 {sorted(extra_top)}")

    depot = data.get("depot")
    if not isinstance(depot, dict):
        warnings.append(f"{fname}: depot 不是对象")
    else:
        dset = set(depot.keys())
        if dset != DEPOT_KEYS:
            warnings.append(f"{fname}: depot 字段不匹配 {sorted(dset)}")

    trucks = data.get("trucks")
    if not isinstance(trucks, list):
        warnings.append(f"{fname}: trucks 不是数组")
    else:
        for idx, t in enumerate(trucks, start=1):
            if not isinstance(t, dict):
                warnings.append(f"{fname}: trucks[{idx}] 不是对象")
                continue
            tset = set(t.keys())
            if tset != TRUCK_KEYS:
                warnings.append(f"{fname}: trucks[{idx}] 字段不匹配 {sorted(tset)}")

    drones = data.get("drones")
    if not isinstance(drones, list):
        warnings.append(f"{fname}: drones 不是数组")
    else:
        for idx, d in enumerate(drones, start=1):
            if not isinstance(d, dict):
                warnings.append(f"{fname}: drones[{idx}] 不是对象")
                continue
            dset = set(d.keys())
            if dset != DRONE_KEYS:
                warnings.append(f"{fname}: drones[{idx}] 字段不匹配 {sorted(dset)}")

    customers = data.get("customers")
    if not isinstance(customers, list):
        warnings.append(f"{fname}: customers 不是数组")
    else:
        for c in customers:
            cid = c.get("customer_id", "?") if isinstance(c, dict) else "?"
            if not isinstance(c, dict):
                warnings.append(f"{fname}: 客户{cid} 不是对象")
                continue
            cset = set(c.keys())
            if cset != CUSTOMER_KEYS:
                warnings.append(f"{fname}: 客户{cid} 字段不匹配 {sorted(cset)}")

    return warnings


def copy_existing_to_stochastic(existing_infos):
    """将已有算例原样复制到 stochastic 目录。"""
    for info in existing_infos:
        src = os.path.join(EXISTING_DIR, info["filename"])
        dst = os.path.join(STO_DIR, info["filename"])
        shutil.copy2(src, dst)


def build_deterministic_from_existing(existing_infos):
    """基于已有算例生成确定性版本：仅 home_probabilities 改为 1.0。"""
    for info in existing_infos:
        src = os.path.join(EXISTING_DIR, info["filename"])
        dst = os.path.join(DET_DIR, info["filename"])
        data = load_json(src)
        customers = data.get("customers", [])
        if isinstance(customers, list):
            for c in customers:
                if isinstance(c, dict):
                    c["home_probabilities"] = HP_ONES.copy()
        dump_json(dst, data)


def scale_from_customers(n_customers):
    if n_customers <= 20:
        return "small"
    if n_customers <= 100:
        return "medium"
    return "large"


def _clustered_coordinates(n_customers, n_pairs, side, rng):
    """聚类分布：簇数=车辆对数；簇中心在中间区域；簇内高斯散布。"""
    cluster_size = n_customers // n_pairs
    remainder = n_customers % n_pairs

    margin = side * 0.15
    centers = []
    for _ in range(n_pairs):
        cx = rng.uniform(margin, side - margin)
        cy = rng.uniform(margin, side - margin)
        centers.append((cx, cy))

    coordinates = []
    for i, (cx, cy) in enumerate(centers):
        size = cluster_size + (1 if i < remainder else 0)
        sigma = side * 0.12
        for _ in range(size):
            x = max(0.0, min(side, rng.gauss(cx, sigma)))
            y = max(0.0, min(side, rng.gauss(cy, sigma)))
            coordinates.append((round(x, 2), round(y, 2)))

    rng.shuffle(coordinates)
    return coordinates


def _uniform_random_coordinates(n_customers, side, rng):
    """均匀随机分布：全域均匀随机。"""
    coordinates = []
    for _ in range(n_customers):
        x = rng.uniform(0.0, side)
        y = rng.uniform(0.0, side)
        coordinates.append((round(x, 2), round(y, 2)))
    rng.shuffle(coordinates)
    return coordinates


def generate_customer_coordinates(n_customers, n_pairs, side, rng, distribution):
    """按分布生成坐标：clustered / random / mixed。"""
    if distribution == "clustered":
        return _clustered_coordinates(n_customers, n_pairs, side, rng)
    if distribution == "random":
        return _uniform_random_coordinates(n_customers, side, rng)
    if distribution == "mixed":
        # 60% 聚类 + 40% 随机
        n_clustered = int(round(n_customers * 0.6))
        n_random = n_customers - n_clustered
        coords = _clustered_coordinates(n_clustered, n_pairs, side, rng) + _uniform_random_coordinates(
            n_random, side, rng
        )
        rng.shuffle(coords)
        return coords
    raise ValueError(f"Unknown distribution: {distribution}")


def generate_home_probs(rng, low_prob_chance=0.2):
    """生成 8 个时间段的在家概率，约 20% 概率落到较低区间。"""
    probs = []
    for _ in range(8):
        if rng.random() < low_prob_chance:
            probs.append(round(rng.uniform(0.35, 0.60), 2))
        else:
            probs.append(round(rng.uniform(0.60, 0.95), 2))
    return probs


def generate_instance(n_customers, n_pairs, seed_number, distribution):
    """按规则生成随机版和确定性版实例。"""
    rng = random.Random(seed_number)

    # 坐标边长（面积与客户数成正比）
    side = round(4.0 * math.sqrt(n_customers / 10.0), 1)

    # delivery/pickup 数量
    n_delivery = round(n_customers * 0.6)
    n_pickup = n_customers - n_delivery

    # 各类型内部 60% 可无人机（weight<=5）
    n_delivery_drone = round(n_delivery * 0.6)
    n_pickup_drone = round(n_pickup * 0.6)

    coordinates = generate_customer_coordinates(n_customers, n_pairs, side, rng, distribution)
    customers = []

    # delivery 客户
    for i in range(n_delivery):
        cid = i + 1
        x, y = coordinates[i]
        if i < n_delivery_drone:
            weight = round(rng.uniform(0.01, 5.0), 2)
        else:
            weight = round(rng.uniform(5.01, 10.0), 2)
        customers.append(
            {
                "customer_id": cid,
                "x": x,
                "y": y,
                "customer_type": "delivery",
                "weight": weight,
                "time_window": [0.0, 480.0],
                "service_time": 10.0,
                "home_probabilities": generate_home_probs(rng),
            }
        )

    # pickup 客户
    for i in range(n_pickup):
        cid = n_delivery + i + 1
        x, y = coordinates[n_delivery + i]
        if i < n_pickup_drone:
            weight = round(rng.uniform(0.01, 5.0), 2)
        else:
            weight = round(rng.uniform(5.01, 10.0), 2)
        customers.append(
            {
                "customer_id": cid,
                "x": x,
                "y": y,
                "customer_type": "pickup",
                "weight": weight,
                "time_window": [0.0, 480.0],
                "service_time": 10.0,
                "home_probabilities": generate_home_probs(rng),
            }
        )

    # 混洗后重新编号，使 delivery/pickup 分布更混合
    rng.shuffle(customers)
    delivery_ids = []
    pickup_ids = []
    for idx, c in enumerate(customers):
        new_id = idx + 1
        c["customer_id"] = new_id
        if c["customer_type"] == "delivery":
            delivery_ids.append(new_id)
        else:
            pickup_ids.append(new_id)

    trucks = [
        {"truck_id": i + 1, "speed": 0.5, "capacity": 100.0}
        for i in range(n_pairs)
    ]
    drones = [
        {
            "drone_id": i + 1,
            "speed": 1.0,
            "capacity": 9.0,
            "max_range_km": 12.0,
            "energy_wh": 500.0,
            "launch_time": 1.0,
            "recovery_time": 1.0,
        }
        for i in range(n_pairs)
    ]

    instance_stochastic = {
        "depot_id": 0,
        "vehicle_pair_count": n_pairs,
        "depot": {"x": 0.0, "y": 0.0},
        "truck_distance_scale": 1.2,
        "drone_distance_scale": 1.0,
        "trucks": trucks,
        "drones": drones,
        "delivery_customers": sorted(delivery_ids),
        "pickup_customers": sorted(pickup_ids),
        "customers": customers,
    }

    instance_deterministic = copy.deepcopy(instance_stochastic)
    for c in instance_deterministic["customers"]:
        c["home_probabilities"] = HP_ONES.copy()

    return instance_stochastic, instance_deterministic


def summarize_instance(filename, data):
    """提取报告所需统计信息。"""
    customers = data.get("customers", [])
    n = len(customers)
    n_delivery = sum(1 for c in customers if c.get("customer_type") == "delivery")
    n_pickup = sum(1 for c in customers if c.get("customer_type") == "pickup")
    n_drone = sum(1 for c in customers if isinstance(c.get("weight"), (int, float)) and c["weight"] <= 5.0)
    n_truck = n - n_drone
    parsed = parse_filename(filename)
    return {
        "filename": filename,
        "scale": parsed["scale"] if parsed else scale_from_customers(n),
        "n_customers": n,
        "n_pairs": data.get("vehicle_pair_count"),
        "distribution": SEED_DISTRIBUTION.get(parsed["seed"]) if parsed else None,
        "delivery": n_delivery,
        "pickup": n_pickup,
        "drone_customers": n_drone,
        "truck_customers": n_truck,
    }


def _deterministic_seed_for_group(n_customers, n_pairs):
    """同一 (n_customers, n_pairs) 组使用固定种子，保证除坐标外一致。"""
    return (n_customers * 1000 + n_pairs * 10 + 7) % 100000


def generate_missing_target_instances(existing_infos):
    """按目标清单补齐缺失实例（skip 已存在的同组合）。"""
    existing_keys = set()
    for info in existing_infos:
        if info["n_customers"] is None:
            continue
        existing_keys.add((info["n_customers"], info["n_pairs"], info["seed"]))

    new_generated = []
    skipped_existing = []

    for n_customers, n_pairs, seed_list in TARGET_SPECS:
        scale = scale_from_customers(n_customers)
        base_seed = _deterministic_seed_for_group(n_customers, n_pairs)
        for seed in seed_list:
            fname = f"{scale}_{n_customers}c_{n_pairs}v_seed{seed}.json"
            key = (n_customers, n_pairs, seed)
            if key in existing_keys:
                skipped_existing.append(fname)
                continue
            distribution = SEED_DISTRIBUTION.get(seed)
            if not distribution:
                raise ValueError(f"Unknown seed -> distribution mapping for seed {seed}")

            # 使用固定 base_seed 生成非坐标字段，保证同组仅坐标不同
            sto, det = generate_instance(n_customers, n_pairs, base_seed, distribution)
            dump_json(os.path.join(STO_DIR, fname), sto)
            dump_json(os.path.join(DET_DIR, fname), det)
            new_generated.append(summarize_instance(fname, sto))

    return new_generated, skipped_existing


def validate_format_for_file(path):
    """执行用户给定的 10 条格式检查。"""
    fname = os.path.basename(path)
    errors = []
    try:
        data = load_json(path)
    except Exception as e:
        return [f"{fname}: JSON 读取失败 -> {e}"]

    required_top = {
        "depot_id",
        "vehicle_pair_count",
        "depot",
        "truck_distance_scale",
        "drone_distance_scale",
        "trucks",
        "drones",
        "delivery_customers",
        "pickup_customers",
        "customers",
    }
    if not required_top.issubset(set(data.keys())):
        errors.append(f"{fname}: 缺少顶层字段 {sorted(required_top - set(data.keys()))}")

    required_cust = {
        "customer_id",
        "x",
        "y",
        "customer_type",
        "weight",
        "time_window",
        "service_time",
        "home_probabilities",
    }
    for c in data.get("customers", []):
        cid = c.get("customer_id", "?")
        if not required_cust.issubset(set(c.keys())):
            errors.append(f"{fname}: 客户{cid} 缺少字段")

    for c in data.get("customers", []):
        cid = c.get("customer_id", "?")
        w = c.get("weight")
        if not isinstance(w, (int, float)) or not (0 < w <= 10.0):
            errors.append(f"{fname}: 客户{cid} weight={w}")

    for c in data.get("customers", []):
        cid = c.get("customer_id", "?")
        tp = c.get("customer_type")
        if tp not in ("delivery", "pickup"):
            errors.append(f"{fname}: 客户{cid} type={tp}")

    for c in data.get("customers", []):
        cid = c.get("customer_id", "?")
        hp = c.get("home_probabilities")
        if not isinstance(hp, list) or len(hp) != 8:
            hp_len = len(hp) if isinstance(hp, list) else "N/A"
            errors.append(f"{fname}: 客户{cid} hp长度={hp_len}")

    all_ids = set(c.get("customer_id") for c in data.get("customers", []))
    listed_ids = set(data.get("delivery_customers", [])) | set(data.get("pickup_customers", []))
    if all_ids != listed_ids:
        errors.append(f"{fname}: ID列表不一致")

    parsed = parse_filename(fname)
    if parsed:
        if parsed["n_customers"] != len(data.get("customers", [])):
            errors.append(f"{fname}: 文件名客户数不一致")
        if parsed["n_pairs"] != data.get("vehicle_pair_count"):
            errors.append(f"{fname}: 文件名车辆对数不一致")

    if len(data.get("trucks", [])) != data.get("vehicle_pair_count"):
        errors.append(f"{fname}: trucks长度不一致")
    if len(data.get("drones", [])) != data.get("vehicle_pair_count"):
        errors.append(f"{fname}: drones长度不一致")

    for c in data.get("customers", []):
        cid = c.get("customer_id", "?")
        if c.get("time_window") != [0.0, 480.0]:
            errors.append(f"{fname}: 客户{cid} 时间窗不是[0,480]")
        if c.get("service_time") != 10.0:
            errors.append(f"{fname}: 客户{cid} 服务时间不是10")

    return errors


def validate_format_all():
    """格式验证：检查 deterministic + stochastic 全部文件。"""
    all_errors = []
    error_files = set()
    total_files = 0

    for d in (DET_DIR, STO_DIR):
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(".json"):
                continue
            total_files += 1
            path = os.path.join(d, name)
            errs = validate_format_for_file(path)
            if errs:
                error_files.add(name)
                all_errors.extend(errs)

    passed_files = total_files - len(error_files)
    return all_errors, passed_files, total_files


def normalized_without_home_probs(data):
    """将实例中 home_probabilities 替换为占位符，用于比较其他字段一致性。"""
    cloned = copy.deepcopy(data)
    for c in cloned.get("customers", []):
        c["home_probabilities"] = "__HP__"
    return cloned


def validate_det_vs_sto_consistency():
    """deterministic vs stochastic 一致性验证。"""
    errors = []
    det_names = sorted([n for n in os.listdir(DET_DIR) if n.lower().endswith(".json")])
    sto_names = sorted([n for n in os.listdir(STO_DIR) if n.lower().endswith(".json")])
    det_set = set(det_names)
    sto_set = set(sto_names)

    for missing in sorted(det_set - sto_set):
        errors.append(f"{missing}: deterministic 存在但 stochastic 缺失")
    for missing in sorted(sto_set - det_set):
        errors.append(f"{missing}: stochastic 存在但 deterministic 缺失")

    checked = 0
    for fname in sorted(det_set & sto_set):
        checked += 1
        det = load_json(os.path.join(DET_DIR, fname))
        sto = load_json(os.path.join(STO_DIR, fname))

        # 除 home_probabilities 外应完全一致
        if normalized_without_home_probs(det) != normalized_without_home_probs(sto):
            errors.append(f"{fname}: 除 home_probabilities 外存在差异")

        # deterministic 必须全部为 1.0
        for c in det.get("customers", []):
            cid = c.get("customer_id", "?")
            hp = c.get("home_probabilities", [])
            if not isinstance(hp, list) or len(hp) != 8 or any(v != 1.0 for v in hp):
                errors.append(f"{fname}: 确定性客户{cid} home_probabilities 非全1.0")

        # stochastic 必须在 [0.35, 0.95]
        for c in sto.get("customers", []):
            cid = c.get("customer_id", "?")
            hp = c.get("home_probabilities", [])
            if not isinstance(hp, list) or len(hp) != 8:
                errors.append(f"{fname}: 随机版客户{cid} home_probabilities 长度异常")
                continue
            for val in hp:
                if not isinstance(val, (int, float)) or not (0.35 <= float(val) <= 0.95):
                    errors.append(f"{fname}: 随机版客户{cid} 概率越界 {val}")
                    break

    passed = checked if not errors else checked - len(set(e.split(":")[0] for e in errors))
    return errors, passed, checked


def validate_counts():
    """数量验证（文件数量、文件名集合、规模档数量阈值）。"""
    errors = []
    det_names = sorted([n for n in os.listdir(DET_DIR) if n.lower().endswith(".json")])
    sto_names = sorted([n for n in os.listdir(STO_DIR) if n.lower().endswith(".json")])
    det_set = set(det_names)
    sto_set = set(sto_names)

    if len(det_names) != len(sto_names):
        errors.append(f"文件数不一致: deterministic={len(det_names)}, stochastic={len(sto_names)}")
    if det_set != sto_set:
        errors.append("两个目录文件名集合不一致")

    # unique 统计（单目录）
    scale_counts = {"small": 0, "medium": 0, "large": 0}
    for name in det_names:
        parsed = parse_filename(name)
        if parsed:
            scale_counts[parsed["scale"]] += 1

    # 按单目录口径检查（deterministic/stochastic 文件名集合一致）
    if scale_counts["small"] < 20:
        errors.append(f"小规模算例总数不足: {scale_counts['small']} < 20")
    if scale_counts["medium"] < 15:
        errors.append(f"中规模算例总数不足: {scale_counts['medium']} < 15")
    if scale_counts["large"] < 18:
        errors.append(f"大规模算例总数不足: {scale_counts['large']} < 18")
    if len(det_names) < 53:
        errors.append(f"总计不足: {len(det_names)} < 53")

    return errors, {
        "det_count": len(det_names),
        "sto_count": len(sto_names),
        "small_count": scale_counts["small"],
        "medium_count": scale_counts["medium"],
        "large_count": scale_counts["large"],
    }


def find_load_instance():
    """查找实例加载函数（优先 load_instance，其次 _load_instance）。"""
    if SOURCE_ROOT not in sys.path:
        sys.path.insert(0, SOURCE_ROOT)

    preferred_candidates = [
        ("spd.core", "load_instance"),
        ("spd.main", "load_instance"),
        ("spd.main", "_load_instance"),
    ]
    for mod_name, fn_name in preferred_candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn, f"{mod_name}.{fn_name}"
        except Exception:
            pass

    candidate_modules = []
    for root, _, files in os.walk(SOURCE_ROOT):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            except Exception:
                continue
            if ("def load_instance" not in txt) and ("def _load_instance" not in txt):
                continue
            rel = os.path.relpath(path, SOURCE_ROOT)
            mod_name = rel[:-3].replace(os.sep, ".")
            if mod_name.endswith(".__init__"):
                mod_name = mod_name[: -len(".__init__")]
            candidate_modules.append(mod_name)

    candidate_modules = sorted(set(candidate_modules), key=lambda s: (0 if s == "spd.core" else 1, s))
    for mod_name in candidate_modules:
        try:
            mod = importlib.import_module(mod_name)
            for fn_name in ("load_instance", "_load_instance"):
                fn = getattr(mod, fn_name, None)
                if callable(fn):
                    return fn, f"{mod_name}.{fn_name}"
        except Exception:
            continue

    raise ImportError("未找到 load_instance / _load_instance 函数")


def validate_loadability():
    """加载验证：逐文件调用 load_instance。"""
    errors = []
    passed = 0
    failed = 0
    module_name = None

    try:
        load_instance, module_name = find_load_instance()
    except Exception as e:
        return [f"查找 load_instance 失败: {e}"], passed, failed, module_name

    for d in (DET_DIR, STO_DIR):
        dname = os.path.basename(d)
        for name in sorted(os.listdir(d)):
            if not name.lower().endswith(".json"):
                continue
            fpath = os.path.join(d, name)
            try:
                load_instance(fpath)
                passed += 1
            except Exception as e:
                failed += 1
                errors.append(f"FAIL: {dname}/{name} -> {e}")

    return errors, passed, failed, module_name


def group_by_scale(items):
    grouped = {"small": [], "medium": [], "large": []}
    for it in items:
        grouped[it["scale"]].append(it)
    for k in grouped:
        grouped[k].sort(key=lambda x: x["filename"])
    return grouped


def generate_report(
    existing_infos,
    structure_warnings,
    new_generated,
    skipped_existing,
    format_errors,
    format_passed,
    format_total,
    consistency_errors,
    consistency_passed,
    consistency_total,
    count_errors,
    count_stats,
    load_errors,
    load_passed,
    load_failed,
    load_module,
):
    """写入 generation_report.txt。"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=== 算例生成报告 ===")
    lines.append(f"生成时间: {now_str}")
    lines.append("")

    lines.append("--- 目录结构 ---")
    lines.append(f"{DET_DIR}\\  → {count_stats['det_count']} 个文件（home_probabilities 全 1.0）")
    lines.append(f"{STO_DIR}\\  → {count_stats['sto_count']} 个文件（home_probabilities 随机）")
    lines.append("")

    lines.append(f"--- 来源：已有算例（从 {EXISTING_DIR} 转换）---")
    lines.append(f"共 {len(existing_infos)} 个，每个生成确定性 + 随机两个版本")
    for info in existing_infos:
        if info["n_customers"] is None:
            lines.append(f"  {info['filename']} | 解析失败")
        else:
            lines.append(
                f"  {info['filename']} | 客户数={info['n_customers']}, 车辆对数={info['n_pairs']}, seed={info['seed']}"
            )
    lines.append("")

    if structure_warnings:
        lines.append("--- 已有算例结构告警 ---")
        for w in structure_warnings:
            lines.append(f"  WARNING: {w}")
        lines.append("")

    grouped_new = group_by_scale(new_generated)
    total_by_scale = {
        "small": count_stats["small_count"],
        "medium": count_stats["medium_count"],
        "large": count_stats["large_count"],
    }

    lines.append("--- 来源：新生成算例 ---")
    for scale_cn, scale_key in (("小规模", "small"), ("中规模", "medium"), ("大规模", "large")):
        new_count = len(grouped_new[scale_key])
        total_count = total_by_scale[scale_key]
        lines.append(f"{scale_cn}（新增 {new_count} 个，总计 {total_count} 个）:")
        if not grouped_new[scale_key]:
            lines.append("  [无新增]")
        else:
            for it in grouped_new[scale_key]:
                lines.append(
                    "  "
                    + f"{it['filename']} | 客户数={it['n_customers']}, 车辆对数={it['n_pairs']}, "
                    + f"分布={it.get('distribution')}, "
                    + f"delivery数={it['delivery']}, pickup数={it['pickup']}, "
                    + f"无人机客户数={it['drone_customers']}, 卡车客户数={it['truck_customers']}"
                )
    lines.append("")

    if skipped_existing:
        lines.append("--- 跳过（已存在同组合） ---")
        for name in sorted(skipped_existing):
            lines.append(f"  {name}")
        lines.append("")

    lines.append("--- 验证结果 ---")
    lines.append(f"格式验证: {format_passed}/{format_total} 通过")
    lines.append(f"一致性验证: {consistency_passed}/{consistency_total} 通过")
    lines.append(f"加载验证: {load_passed}/{load_passed + load_failed} 通过")
    if load_module:
        lines.append(f"load_instance 来源: {load_module}")
    lines.append("")

    if format_errors:
        lines.append("格式验证错误详情:")
        for e in format_errors:
            lines.append(f"  {e}")
        lines.append("")
    if consistency_errors:
        lines.append("一致性验证错误详情:")
        for e in consistency_errors:
            lines.append(f"  {e}")
        lines.append("")
    if count_errors:
        lines.append("数量验证错误详情:")
        for e in count_errors:
            lines.append(f"  {e}")
        lines.append("")
    if load_errors:
        lines.append("加载验证错误详情:")
        for e in load_errors:
            lines.append(f"  {e}")
        lines.append("")

    # 新算例统计（平均比例）
    if new_generated:
        total_new_customers = sum(x["n_customers"] for x in new_generated)
        total_new_delivery = sum(x["delivery"] for x in new_generated)
        total_new_pickup = sum(x["pickup"] for x in new_generated)
        total_new_drone = sum(x["drone_customers"] for x in new_generated)
        delivery_pct = (100.0 * total_new_delivery / total_new_customers) if total_new_customers else 0.0
        pickup_pct = (100.0 * total_new_pickup / total_new_customers) if total_new_customers else 0.0
        drone_pct = (100.0 * total_new_drone / total_new_customers) if total_new_customers else 0.0
    else:
        delivery_pct = pickup_pct = drone_pct = 0.0

    lines.append("--- 统计 ---")
    lines.append(f"确定性目录总文件数: {count_stats['det_count']}")
    lines.append(f"随机目录总文件数: {count_stats['sto_count']}")
    lines.append(f"delivery:pickup 比例（新算例平均）: {delivery_pct:.2f}%:{pickup_pct:.2f}%")
    lines.append(f"无人机客户占比（weight≤5，新算例平均）: {drone_pct:.2f}%")
    lines.append("坐标范围示例: 10c→[0,4.0], 100c→[0,12.6], 300c→[0,21.9]")
    lines.append("depot 坐标: (0.0, 0.0)")
    lines.append("home_probabilities 格式: 长度8数组，随机值[0.35,0.95]")
    lines.append("seed 含义: seed1=clustered, seed2=random, seed3=mixed（仅坐标分布不同）")
    lines.append("")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("=== 第一步：盘点已有算例 ===")
    existing_infos = get_existing_inventory()
    print(f"已有文件数: {len(existing_infos)}")
    for info in existing_infos:
        print(
            f"- {info['filename']} | n={info['n_customers']} | v={info['n_pairs']} | seed={info['seed']}"
        )

    print("\n=== 第一步：结构校验 ===")
    structure_warnings = []
    for info in existing_infos:
        path = os.path.join(EXISTING_DIR, info["filename"])
        try:
            data = load_json(path)
            structure_warnings.extend(validate_structure_against_template(data, info["filename"]))
        except Exception as e:
            structure_warnings.append(f"{info['filename']}: 读取失败 -> {e}")
    if structure_warnings:
        print(f"结构告警数: {len(structure_warnings)}")
        for w in structure_warnings:
            print(f"WARNING: {w}")
    else:
        print("结构校验通过（无告警）")

    print("\n=== 第二步：生成两套算例 ===")
    ensure_output_dirs()
    clear_json_files(DET_DIR)
    clear_json_files(STO_DIR)

    # 已有算例：随机版原样复制，确定性版仅改 home_probabilities
    copy_existing_to_stochastic(existing_infos)
    build_deterministic_from_existing(existing_infos)

    # 目标清单：跳过已存在同 (n,v,seed)
    new_generated, skipped_existing = generate_missing_target_instances(existing_infos)
    print(f"新增文件数（每个目录）: {len(new_generated)}")
    print(f"跳过已存在组合数: {len(skipped_existing)}")

    print("\n=== 第三步：验证 ===")
    format_errors, format_passed, format_total = validate_format_all()
    consistency_errors, consistency_passed, consistency_total = validate_det_vs_sto_consistency()
    count_errors, count_stats = validate_counts()
    load_errors, load_passed, load_failed, load_module = validate_loadability()

    print(f"格式验证: {format_passed}/{format_total} 通过")
    print(f"一致性验证: {consistency_passed}/{consistency_total} 通过")
    print(f"加载验证: {load_passed}/{load_passed + load_failed} 通过")
    if load_module:
        print(f"load_instance 模块: {load_module}")

    print("\n=== 第四步：生成报告 ===")
    generate_report(
        existing_infos=existing_infos,
        structure_warnings=structure_warnings,
        new_generated=new_generated,
        skipped_existing=skipped_existing,
        format_errors=format_errors,
        format_passed=format_passed,
        format_total=format_total,
        consistency_errors=consistency_errors,
        consistency_passed=consistency_passed,
        consistency_total=consistency_total,
        count_errors=count_errors,
        count_stats=count_stats,
        load_errors=load_errors,
        load_passed=load_passed,
        load_failed=load_failed,
        load_module=load_module,
    )
    print(f"报告已写入: {REPORT_PATH}")

    print("\n=== 第五步：保存生成脚本 ===")
    src_script = os.path.abspath(__file__)
    # 将完整脚本保存到 D:\examples\generate_instances.py
    if os.path.abspath(src_script) != os.path.abspath(SCRIPT_OUTPUT_PATH):
        shutil.copy2(src_script, SCRIPT_OUTPUT_PATH)
    print(f"脚本已保存: {SCRIPT_OUTPUT_PATH}")

    fatal_errors = []
    fatal_errors.extend(format_errors)
    fatal_errors.extend(consistency_errors)
    fatal_errors.extend(count_errors)
    fatal_errors.extend(load_errors)

    if fatal_errors:
        print("\n存在验证失败，请查看报告中的错误详情。")
        return 1

    print("\n全部生成与验证完成。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"脚本异常退出: {exc}")
        raise
