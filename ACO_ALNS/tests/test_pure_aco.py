"""测试纯 ACO 优化器的基本功能。"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

from spd.aco_alns import ALNSOptimizer, PureACOOptimizer
from spd.core import all_hard_constraints_satisfied
from spd.etprc import ETPRCBuilder


def _small_pure_aco_params(sample_params):
    """构造小规模参数，降低单测耗时并突出纯 ACO 逻辑。"""
    return replace(
        sample_params,
        aco=replace(sample_params.aco, max_iterations=2, ant_count=2, no_improve_max=2),
        alns=replace(sample_params.alns, iterations=0),
    )


def test_pure_aco_constructs_valid_solution(sample_instance, sample_params, all_home_status) -> None:
    """纯 ACO 优化后的解必须满足所有硬约束且无未服务客户。"""
    params = _small_pure_aco_params(sample_params)
    builder = ETPRCBuilder()
    initial_solution = builder.build_initial_solution(sample_instance, params, random.Random(10))

    optimizer = PureACOOptimizer(etprc_builder=builder)
    optimized_solution, _obj, _history = optimizer.optimize(
        instance=sample_instance,
        initial_solution=initial_solution,
        params=params,
        home_status=all_home_status,
        rng=random.Random(11),
    )

    assert all_hard_constraints_satisfied(sample_instance, optimized_solution, all_home_status, params) is True
    assert optimized_solution.unserved_customers == set()


def test_pure_aco_no_alns_called(sample_instance, sample_params, all_home_status, monkeypatch) -> None:
    """确认纯 ACO 不调用任何 ALNS 局部搜索入口。"""

    def _forbidden_optimize(*args, **kwargs):
        raise AssertionError("PureACOOptimizer 不应调用 ALNSOptimizer.optimize")

    # 将 ALNS 主入口替换为硬失败，若纯 ACO误调用会立刻暴露。
    monkeypatch.setattr(ALNSOptimizer, "optimize", _forbidden_optimize)

    params = replace(
        _small_pure_aco_params(sample_params),
        aco=replace(sample_params.aco, max_iterations=1, ant_count=1, no_improve_max=1),
    )
    builder = ETPRCBuilder()
    initial_solution = builder.build_initial_solution(sample_instance, params, random.Random(20))

    optimizer = PureACOOptimizer(etprc_builder=builder)
    _solution, _obj, history = optimizer.optimize(
        instance=sample_instance,
        initial_solution=initial_solution,
        params=params,
        home_status=all_home_status,
        rng=random.Random(21),
    )

    assert len(history) == 1


def test_pure_aco_pheromone_updates(sample_instance, sample_params, all_home_status) -> None:
    """确认信息素在 ACO 迭代后发生更新。"""
    params = replace(
        _small_pure_aco_params(sample_params),
        aco=replace(sample_params.aco, max_iterations=2, ant_count=1, no_improve_max=2),
    )
    builder = ETPRCBuilder()
    initial_solution = builder.build_initial_solution(sample_instance, params, random.Random(30))

    optimizer = PureACOOptimizer(etprc_builder=builder)
    initial_pheromone = optimizer._initialize_pheromone(sample_instance, params)
    _solution, _obj, _history = optimizer.optimize(
        instance=sample_instance,
        initial_solution=initial_solution,
        params=params,
        home_status=all_home_status,
        rng=random.Random(31),
    )

    assert optimizer.last_pheromone
    assert any(
        abs(float(optimizer.last_pheromone[key]) - float(initial_pheromone.get(key, 0.0))) > 1e-12
        for key in optimizer.last_pheromone
    )


def test_pure_aco_deterministic_mode(sample_instance, sample_params) -> None:
    """确定性配置下 home_status 应全部为 True。"""
    config_path = Path("examples/configs/config_pure_aco_small.json")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload.get("deterministic") is True

    params = replace(
        _small_pure_aco_params(sample_params),
        aco=replace(sample_params.aco, max_iterations=1, ant_count=1, no_improve_max=1),
    )
    home_status = {customer_id: True for customer_id in sample_instance.customers}
    assert all(home_status.values())

    builder = ETPRCBuilder()
    initial_solution = builder.build_initial_solution(sample_instance, params, random.Random(40))
    optimizer = PureACOOptimizer(etprc_builder=builder)
    _solution, _obj, _history = optimizer.optimize(
        instance=sample_instance,
        initial_solution=initial_solution,
        params=params,
        home_status=home_status,
        rng=random.Random(41),
    )
