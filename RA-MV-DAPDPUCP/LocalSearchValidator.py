import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from typing import Dict, List, Tuple
import pandas as pd


class LocalSearchValidator:
    """
    局部搜索有效性验证工具
    用于检测和分析局部搜索的实际效果
    """

    def __init__(self, dynamic_optimizer):
        self.dyn_opt = dynamic_optimizer
        self.validation_results = {
            'trigger_analysis': {},
            'operator_performance': {},
            'improvement_tracking': {},
            'time_analysis': {},
            'convergence_analysis': {}
        }

    def run_comprehensive_validation(self, num_tests: int = 1000):
        """
        运行完整的局部搜索验证测试
        """
        print("\n" + "=" * 80)
        print("🔍 开始局部搜索有效性综合验证")
        print("=" * 80)

        # 1. 触发机制验证
        self.validate_trigger_mechanism()

        # 2. 单算子效果测试
        self.validate_individual_operators()

        # 3. 改进追踪测试
        self.track_improvement_patterns(num_tests)

        # 4. 时间效率分析
        self.analyze_time_efficiency()

        # 5. 收敛性分析
        self.analyze_convergence_behavior()

        # 6. 对比测试（有/无局部搜索）
        self.comparative_analysis(num_tests)

        # 7. 生成综合报告
        self.generate_validation_report()

        return self.validation_results

    def validate_trigger_mechanism(self):
        """
        验证局部搜索触发机制是否正常工作
        """
        print("\n📌 测试1: 局部搜索触发机制验证")
        print("-" * 40)

        trigger_tests = []
        test_scenarios = [
            ("成本改进场景", 1000, 900),  # 有改进
            ("成本持平场景", 1000, 1000),  # 无改进
            ("小幅恶化场景", 1000, 1050),  # 5%恶化
            ("大幅恶化场景", 1000, 1400),  # 40%恶化
        ]

        # 保存原始设置
        original_theta = self.dyn_opt.theta
        original_count = self.dyn_opt.ls_operation_count

        for scenario_name, cost_before, cost_after in test_scenarios:
            # 测试每种场景
            for truck_id in range(len(self.dyn_opt.TRUCK_Routes)):
                should_trigger, reason = self.dyn_opt.enhanced_local_search_trigger(
                    truck_id, cost_before, cost_after
                )

                trigger_tests.append({
                    'scenario': scenario_name,
                    'cost_before': cost_before,
                    'cost_after': cost_after,
                    'improvement_rate': (cost_before - cost_after) / cost_before * 100,
                    'triggered': should_trigger,
                    'reason': reason
                })

                print(f"  {scenario_name}: {cost_before}→{cost_after} | "
                      f"触发={should_trigger} | 原因={reason}")
                break  # 只测试第一个车辆对

        # 恢复原始设置
        self.dyn_opt.theta = original_theta
        self.dyn_opt.ls_operation_count = original_count

        # 分析触发率
        trigger_rate = sum(1 for t in trigger_tests if t['triggered']) / len(trigger_tests) * 100
        print(f"\n  📊 触发率: {trigger_rate:.1f}%")

        self.validation_results['trigger_analysis'] = {
            'tests': trigger_tests,
            'trigger_rate': trigger_rate
        }

        return trigger_rate > 50  # 期望至少50%的场景触发

    def validate_individual_operators(self):
        """
        验证每个局部搜索算子的独立效果
        """
        print("\n📌 测试2: 局部搜索算子独立效果验证")
        print("-" * 40)

        operators = ['intra_move', 'intra_swap', 'intra_2opt']
        operator_stats = {}

        for truck_id in range(min(3, len(self.dyn_opt.TRUCK_Routes))):  # 测试前3个车辆对
            if len(self.dyn_opt.TRUCK_Routes[truck_id].Troute) < 4:
                continue

            print(f"\n  车辆对{truck_id}测试:")

            for op_name in operators:
                # 备份当前状态
                backup_truck = copy.deepcopy(self.dyn_opt.TRUCK_Routes[truck_id])
                backup_drone = copy.deepcopy(self.dyn_opt.DRONE_Routes[truck_id])

                # 记录操作前成本
                cost_before = self.dyn_opt.cost_single_vehicle(truck_id)

                # 执行单个算子多次测试
                successes = 0
                improvements = []
                execution_times = []

                for _ in range(10):  # 每个算子测试10次
                    start_time = time.time()

                    try:
                        if op_name == 'intra_move':
                            success = self.dyn_opt._intra_move_within_vehicle(truck_id)
                        elif op_name == 'intra_swap':
                            success = self.dyn_opt._intra_swap_within_vehicle(truck_id)
                        elif op_name == 'intra_2opt':
                            success = self.dyn_opt._intra_2opt_within_vehicle(truck_id)
                        else:
                            success = False

                        execution_time = time.time() - start_time
                        execution_times.append(execution_time)

                        if success:
                            successes += 1
                            cost_after = self.dyn_opt.cost_single_vehicle(truck_id)
                            improvement = cost_before - cost_after
                            improvements.append(improvement)

                            # 恢复状态继续测试
                            self.dyn_opt.TRUCK_Routes[truck_id] = copy.deepcopy(backup_truck)
                            self.dyn_opt.DRONE_Routes[truck_id] = copy.deepcopy(backup_drone)

                    except Exception as e:
                        print(f"    ⚠️ {op_name}执行异常: {e}")

                # 统计该算子性能
                if op_name not in operator_stats:
                    operator_stats[op_name] = {
                        'success_count': 0,
                        'total_tests': 0,
                        'improvements': [],
                        'execution_times': []
                    }

                operator_stats[op_name]['success_count'] += successes
                operator_stats[op_name]['total_tests'] += 10
                operator_stats[op_name]['improvements'].extend(improvements)
                operator_stats[op_name]['execution_times'].extend(execution_times)

                # 输出单算子统计
                success_rate = successes / 10 * 100
                avg_improvement = np.mean(improvements) if improvements else 0
                avg_time = np.mean(execution_times) * 1000  # 转换为毫秒

                print(f"    {op_name:15} | 成功率: {success_rate:5.1f}% | "
                      f"平均改进: {avg_improvement:8.2f} | "
                      f"平均耗时: {avg_time:6.2f}ms")

                # 恢复原始状态
                self.dyn_opt.TRUCK_Routes[truck_id] = backup_truck
                self.dyn_opt.DRONE_Routes[truck_id] = backup_drone

        # 计算总体统计
        for op_name, stats in operator_stats.items():
            if stats['total_tests'] > 0:
                stats['success_rate'] = stats['success_count'] / stats['total_tests'] * 100
                stats['avg_improvement'] = np.mean(stats['improvements']) if stats['improvements'] else 0
                stats['avg_execution_time'] = np.mean(stats['execution_times']) if stats['execution_times'] else 0

        self.validation_results['operator_performance'] = operator_stats

        # 判断是否有效
        total_success_rate = sum(s['success_rate'] for s in operator_stats.values()) / len(operator_stats)
        print(f"\n  📊 算子总体成功率: {total_success_rate:.1f}%")

        return total_success_rate > 30  # 期望至少30%成功率

    def track_improvement_patterns(self, num_iterations: int = 10):
        """
        追踪局部搜索的改进模式
        """
        print("\n📌 测试3: 改进模式追踪")
        print("-" * 40)

        improvement_history = []
        cost_history = []

        for iteration in range(num_iterations):
            print(f"\n  迭代 {iteration + 1}/{num_iterations}:")

            # 随机选择一个车辆对
            truck_id = np.random.randint(0, len(self.dyn_opt.TRUCK_Routes))

            # 记录初始成本
            initial_cost = self.dyn_opt.cost()
            cost_before_ls = initial_cost

            # 执行局部搜索
            if self.dyn_opt.enable_local_search:
                try:
                    # 备份状态
                    backup_state = self._backup_solution_state()

                    # 执行局部搜索
                    cost_after_ls = self.dyn_opt.local_search(truck_id, cost_before_ls)

                    improvement = cost_before_ls - cost_after_ls
                    improvement_rate = improvement / cost_before_ls * 100 if cost_before_ls > 0 else 0

                    improvement_history.append({
                        'iteration': iteration,
                        'truck_id': truck_id,
                        'cost_before': cost_before_ls,
                        'cost_after': cost_after_ls,
                        'improvement': improvement,
                        'improvement_rate': improvement_rate
                    })

                    cost_history.append(cost_after_ls)

                    print(f"    车辆对{truck_id}: {cost_before_ls:.2f} → {cost_after_ls:.2f} "
                          f"(改进: {improvement:.2f}, {improvement_rate:.2f}%)")

                    # 恢复状态（为了下次测试的独立性）
                    self._restore_solution_state(backup_state)

                except Exception as e:
                    print(f"    ⚠️ 局部搜索执行失败: {e}")
                    cost_history.append(cost_before_ls)

        # 分析改进模式
        if improvement_history:
            total_improvements = sum(h['improvement'] for h in improvement_history)
            positive_improvements = sum(1 for h in improvement_history if h['improvement'] > 0)
            improvement_rate = positive_improvements / len(improvement_history) * 100
            avg_improvement = np.mean([h['improvement'] for h in improvement_history])

            print(f"\n  📊 改进模式分析:")
            print(f"     总改进: {total_improvements:.2f}")
            print(f"     改进次数: {positive_improvements}/{len(improvement_history)}")
            print(f"     改进率: {improvement_rate:.1f}%")
            print(f"     平均改进: {avg_improvement:.2f}")

            self.validation_results['improvement_tracking'] = {
                'history': improvement_history,
                'total_improvement': total_improvements,
                'improvement_rate': improvement_rate,
                'avg_improvement': avg_improvement
            }

        return improvement_rate > 20  # 期望至少20%的改进率

    def analyze_time_efficiency(self):
        """
        分析局部搜索的时间效率
        """
        print("\n📌 测试4: 时间效率分析")
        print("-" * 40)

        time_stats = {
            'with_ls': [],
            'without_ls': []
        }

        # 测试有局部搜索的情况
        print("\n  测试有局部搜索的执行时间:")
        self.dyn_opt.enable_local_search = True

        for i in range(5):  # 测试5次
            truck_id = i % len(self.dyn_opt.TRUCK_Routes)

            start_time = time.time()
            try:
                # 执行一次完整的更新流程（包含局部搜索）
                backup_state = self._backup_solution_state()
                cost_before = self.dyn_opt.cost()

                # 模拟局部搜索调用
                if len(self.dyn_opt.TRUCK_Routes[truck_id].Troute) > 3:
                    self.dyn_opt.local_search(truck_id, cost_before)

                execution_time = time.time() - start_time
                time_stats['with_ls'].append(execution_time)

                print(f"    测试{i + 1}: {execution_time * 1000:.2f}ms")

                # 恢复状态
                self._restore_solution_state(backup_state)

            except Exception as e:
                print(f"    ⚠️ 测试{i + 1}失败: {e}")

        # 测试无局部搜索的情况
        print("\n  测试无局部搜索的执行时间:")
        self.dyn_opt.enable_local_search = False

        for i in range(5):
            truck_id = i % len(self.dyn_opt.TRUCK_Routes)

            start_time = time.time()
            try:
                # 执行一次更新流程（不包含局部搜索）
                backup_state = self._backup_solution_state()

                # 只执行基本操作
                if len(self.dyn_opt.TRUCK_Routes[truck_id].Troute) > 3:
                    # 模拟一些基本操作
                    _ = self.dyn_opt.cost_single_vehicle(truck_id)

                execution_time = time.time() - start_time
                time_stats['without_ls'].append(execution_time)

                print(f"    测试{i + 1}: {execution_time * 1000:.2f}ms")

                # 恢复状态
                self._restore_solution_state(backup_state)

            except Exception as e:
                print(f"    ⚠️ 测试{i + 1}失败: {e}")

        # 恢复原始设置
        self.dyn_opt.enable_local_search = True

        # 计算统计信息
        if time_stats['with_ls'] and time_stats['without_ls']:
            avg_with_ls = np.mean(time_stats['with_ls']) * 1000
            avg_without_ls = np.mean(time_stats['without_ls']) * 1000
            time_overhead = avg_with_ls - avg_without_ls
            overhead_rate = time_overhead / avg_without_ls * 100 if avg_without_ls > 0 else 0

            print(f"\n  📊 时间效率分析:")
            print(f"     有局部搜索平均耗时: {avg_with_ls:.2f}ms")
            print(f"     无局部搜索平均耗时: {avg_without_ls:.2f}ms")
            print(f"     额外开销: {time_overhead:.2f}ms ({overhead_rate:.1f}%)")

            self.validation_results['time_analysis'] = {
                'avg_with_ls': avg_with_ls,
                'avg_without_ls': avg_without_ls,
                'overhead': time_overhead,
                'overhead_rate': overhead_rate
            }

            return overhead_rate < 200  # 期望开销不超过200%

        return False

    def analyze_convergence_behavior(self):
        """
        分析局部搜索的收敛行为
        """
        print("\n📌 测试5: 收敛行为分析")
        print("-" * 40)

        convergence_data = []

        # 选择一个车辆对进行深入分析
        truck_id = 0
        if len(self.dyn_opt.TRUCK_Routes[truck_id].Troute) < 4:
            print("  ⚠️ 车辆对0路径太短，跳过收敛分析")
            return False

        # 备份初始状态
        initial_backup = self._backup_solution_state()

        # 设置局部搜索参数
        original_max_no_improve = self.dyn_opt.local_search_max_no_improve
        self.dyn_opt.local_search_max_no_improve = 20  # 增加迭代次数以观察收敛

        # 执行多轮局部搜索，记录每轮的成本
        print(f"\n  对车辆对{truck_id}执行连续局部搜索:")

        current_cost = self.dyn_opt.cost()
        for round_num in range(10):  # 执行10轮
            round_start_cost = current_cost

            try:
                # 执行局部搜索
                new_cost = self.dyn_opt.local_search(truck_id, current_cost)
                improvement = current_cost - new_cost

                convergence_data.append({
                    'round': round_num + 1,
                    'start_cost': round_start_cost,
                    'end_cost': new_cost,
                    'improvement': improvement
                })

                print(f"    第{round_num + 1}轮: {round_start_cost:.2f} → {new_cost:.2f} "
                      f"(改进: {improvement:.2f})")

                current_cost = new_cost

                # 如果没有改进，说明已收敛
                if improvement < 0.001:
                    print(f"    ✅ 在第{round_num + 1}轮达到收敛")
                    break

            except Exception as e:
                print(f"    ⚠️ 第{round_num + 1}轮执行失败: {e}")
                break

        # 恢复设置
        self.dyn_opt.local_search_max_no_improve = original_max_no_improve
        self._restore_solution_state(initial_backup)

        # 分析收敛特性
        if convergence_data:
            improvements = [d['improvement'] for d in convergence_data]
            converged_round = next((i for i, imp in enumerate(improvements) if imp < 0.001), -1)

            print(f"\n  📊 收敛分析:")
            print(f"     总轮数: {len(convergence_data)}")
            print(f"     收敛轮次: {converged_round + 1 if converged_round >= 0 else '未收敛'}")
            print(f"     总改进: {sum(improvements):.2f}")
            print(f"     改进递减率: {self._calculate_decay_rate(improvements):.2f}%")

            self.validation_results['convergence_analysis'] = {
                'data': convergence_data,
                'converged_at': converged_round + 1 if converged_round >= 0 else None,
                'total_improvement': sum(improvements)
            }

            return converged_round >= 0  # 期望能够收敛

        return False

    def comparative_analysis(self, num_tests: int = 10):
        """
        有/无局部搜索的对比分析
        """
        print("\n📌 测试6: 有/无局部搜索对比分析")
        print("-" * 40)

        results_with_ls = []
        results_without_ls = []

        # 备份初始状态
        initial_backup = self._backup_solution_state()

        print("\n  执行对比测试:")

        for test_id in range(num_tests):
            # 每次测试都从相同的初始状态开始
            self._restore_solution_state(initial_backup)

            # 随机选择一个车辆对和客户进行测试
            truck_id = test_id % len(self.dyn_opt.TRUCK_Routes)
            if len(self.dyn_opt.get_vehicle_customers(truck_id)) < 3:
                continue

            customers = list(self.dyn_opt.get_vehicle_customers(truck_id))
            if customers:
                customer_id = np.random.choice(customers)
            else:
                continue

            # 测试有局部搜索的情况
            self.dyn_opt.enable_local_search = True
            self.dyn_opt.ls_operation_count = 0  # 重置计数器

            cost_before = self.dyn_opt.cost()

            try:
                # 模拟一次路径更新（会触发局部搜索）
                self.dyn_opt.update_route(1, customer_id, ['tk', truck_id])
                cost_with_ls = self.dyn_opt.cost()

                results_with_ls.append({
                    'test_id': test_id,
                    'cost_before': cost_before,
                    'cost_after': cost_with_ls,
                    'improvement': cost_before - cost_with_ls
                })

            except Exception as e:
                print(f"    ⚠️ 测试{test_id + 1}(有LS)失败: {e}")

            # 恢复初始状态
            self._restore_solution_state(initial_backup)

            # 测试无局部搜索的情况
            self.dyn_opt.enable_local_search = False

            try:
                # 执行相同的路径更新（不会触发局部搜索）
                self.dyn_opt.update_route(1, customer_id, ['tk', truck_id])
                cost_without_ls = self.dyn_opt.cost()

                results_without_ls.append({
                    'test_id': test_id,
                    'cost_before': cost_before,
                    'cost_after': cost_without_ls,
                    'improvement': cost_before - cost_without_ls
                })

            except Exception as e:
                print(f"    ⚠️ 测试{test_id + 1}(无LS)失败: {e}")

            # 恢复初始状态
            self._restore_solution_state(initial_backup)

        # 恢复局部搜索设置
        self.dyn_opt.enable_local_search = True

        # 分析对比结果
        if results_with_ls and results_without_ls:
            avg_improvement_with_ls = np.mean([r['improvement'] for r in results_with_ls])
            avg_improvement_without_ls = np.mean([r['improvement'] for r in results_without_ls])

            improvement_difference = avg_improvement_with_ls - avg_improvement_without_ls

            print(f"\n  📊 对比分析结果:")
            print(f"     有局部搜索平均改进: {avg_improvement_with_ls:.2f}")
            print(f"     无局部搜索平均改进: {avg_improvement_without_ls:.2f}")
            print(f"     局部搜索额外贡献: {improvement_difference:.2f}")

            # 统计显著改进的比例
            significant_improvements = sum(
                1 for w, wo in zip(results_with_ls, results_without_ls)
                if w['improvement'] > wo['improvement'] + 0.01
            )
            improvement_rate = significant_improvements / len(results_with_ls) * 100

            print(f"     显著改进比例: {improvement_rate:.1f}%")

            return improvement_difference > 0  # 期望局部搜索带来正面贡献

        return False

    def generate_validation_report(self):
        """
        生成综合验证报告
        """
        print("\n" + "=" * 80)
        print("📋 局部搜索有效性验证报告")
        print("=" * 80)

        # 1. 触发机制评估
        if 'trigger_analysis' in self.validation_results:
            trigger_rate = self.validation_results['trigger_analysis']['trigger_rate']
            verdict = "✅ 正常" if trigger_rate > 50 else "❌ 异常"
            print(f"\n1. 触发机制: {verdict}")
            print(f"   - 触发率: {trigger_rate:.1f}%")

        # 2. 算子性能评估
        if 'operator_performance' in self.validation_results:
            operator_stats = self.validation_results['operator_performance']
            print(f"\n2. 算子性能:")
            for op_name, stats in operator_stats.items():
                if 'success_rate' in stats:
                    print(f"   - {op_name}: 成功率={stats['success_rate']:.1f}%, "
                          f"平均改进={stats['avg_improvement']:.2f}")

        # 3. 改进效果评估
        if 'improvement_tracking' in self.validation_results:
            tracking = self.validation_results['improvement_tracking']
            verdict = "✅ 有效" if tracking.get('improvement_rate', 0) > 20 else "⚠️ 效果有限"
            print(f"\n3. 改进效果: {verdict}")
            print(f"   - 改进率: {tracking.get('improvement_rate', 0):.1f}%")
            print(f"   - 平均改进: {tracking.get('avg_improvement', 0):.2f}")

        # 4. 时间效率评估
        if 'time_analysis' in self.validation_results:
            time_analysis = self.validation_results['time_analysis']
            overhead = time_analysis.get('overhead_rate', 0)
            verdict = "✅ 可接受" if overhead < 200 else "⚠️ 开销较大"
            print(f"\n4. 时间效率: {verdict}")
            print(f"   - 额外开销: {overhead:.1f}%")

        # 5. 收敛性评估
        if 'convergence_analysis' in self.validation_results:
            convergence = self.validation_results['convergence_analysis']
            converged = convergence.get('converged_at')
            verdict = "✅ 良好" if converged else "⚠️ 未收敛"
            print(f"\n5. 收敛性: {verdict}")
            if converged:
                print(f"   - 收敛轮次: {converged}")

        # 总体评估
        print("\n" + "=" * 80)
        print("📊 总体评估:")

        effectiveness_score = self._calculate_effectiveness_score()

        if effectiveness_score >= 80:
            print("   ✅ 局部搜索模块工作正常且有效")
        elif effectiveness_score >= 50:
            print("   ⚠️ 局部搜索模块基本有效，但有改进空间")
        else:
            print("   ❌ 局部搜索模块效果不佳，需要优化")

        print(f"   综合得分: {effectiveness_score:.1f}/100")
        print("=" * 80)

    def _backup_solution_state(self):
        """备份当前解状态"""
        return {
            'truck_routes': copy.deepcopy(self.dyn_opt.TRUCK_Routes),
            'drone_routes': copy.deepcopy(self.dyn_opt.DRONE_Routes),
            'customers': copy.deepcopy(self.dyn_opt.customers)
        }

    def _restore_solution_state(self, backup):
        """恢复解状态"""
        self.dyn_opt.TRUCK_Routes = copy.deepcopy(backup['truck_routes'])
        self.dyn_opt.DRONE_Routes = copy.deepcopy(backup['drone_routes'])
        self.dyn_opt.customers = copy.deepcopy(backup['customers'])

    def _calculate_decay_rate(self, improvements):
        """计算改进递减率"""
        if len(improvements) < 2:
            return 0

        decay_rates = []
        for i in range(1, len(improvements)):
            if improvements[i - 1] > 0:
                rate = (improvements[i - 1] - improvements[i]) / improvements[i - 1] * 100
                decay_rates.append(rate)

        return np.mean(decay_rates) if decay_rates else 0

    def _calculate_effectiveness_score(self):
        """计算局部搜索有效性综合得分"""
        score = 0
        weights = {
            'trigger': 20,
            'operators': 20,
            'improvement': 25,
            'time': 15,
            'convergence': 20
        }

        # 触发机制得分
        if 'trigger_analysis' in self.validation_results:
            trigger_rate = self.validation_results['trigger_analysis']['trigger_rate']
            score += weights['trigger'] * min(trigger_rate / 70, 1)

        # 算子性能得分
        if 'operator_performance' in self.validation_results:
            operator_stats = self.validation_results['operator_performance']
            if operator_stats:
                avg_success = np.mean([s.get('success_rate', 0) for s in operator_stats.values()])
                score += weights['operators'] * min(avg_success / 40, 1)

        # 改进效果得分
        if 'improvement_tracking' in self.validation_results:
            improvement_rate = self.validation_results['improvement_tracking'].get('improvement_rate', 0)
            score += weights['improvement'] * min(improvement_rate / 30, 1)

        # 时间效率得分
        if 'time_analysis' in self.validation_results:
            overhead = self.validation_results['time_analysis'].get('overhead_rate', 999)
            if overhead < 200:
                score += weights['time'] * (1 - min(overhead / 200, 1))

        # 收敛性得分
        if 'convergence_analysis' in self.validation_results:
            if self.validation_results['convergence_analysis'].get('converged_at'):
                score += weights['convergence']

        return score


def visualize_local_search_performance(validator: LocalSearchValidator):
    """
    可视化局部搜索性能分析结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('局部搜索性能分析可视化', fontsize=16)

    # 1. 触发率条形图
    ax1 = axes[0, 0]
    if 'trigger_analysis' in validator.validation_results:
        trigger_data = validator.validation_results['trigger_analysis']['tests']
        scenarios = list(set(t['scenario'] for t in trigger_data))
        trigger_rates = [
            sum(1 for t in trigger_data if t['scenario'] == s and t['triggered']) /
            sum(1 for t in trigger_data if t['scenario'] == s) * 100
            for s in scenarios
        ]
        ax1.bar(range(len(scenarios)), trigger_rates)
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels(scenarios, rotation=45, ha='right')
        ax1.set_ylabel('触发率 (%)')
        ax1.set_title('不同场景触发率')

    # 2. 算子成功率
    ax2 = axes[0, 1]
    if 'operator_performance' in validator.validation_results:
        operator_stats = validator.validation_results['operator_performance']
        operators = list(operator_stats.keys())
        success_rates = [operator_stats[op].get('success_rate', 0) for op in operators]
        ax2.bar(operators, success_rates)
        ax2.set_ylabel('成功率 (%)')
        ax2.set_title('算子成功率对比')

    # 3. 改进趋势
    ax3 = axes[0, 2]
    if 'improvement_tracking' in validator.validation_results:
        history = validator.validation_results['improvement_tracking'].get('history', [])
        if history:
            iterations = [h['iteration'] for h in history]
            improvements = [h['improvement'] for h in history]
            ax3.plot(iterations, improvements, 'b-o')
            ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax3.set_xlabel('迭代次数')
            ax3.set_ylabel('改进值')
            ax3.set_title('改进趋势')

    # 4. 时间开销对比
    ax4 = axes[1, 0]
    if 'time_analysis' in validator.validation_results:
        time_data = validator.validation_results['time_analysis']
        categories = ['无局部搜索', '有局部搜索']
        times = [time_data.get('avg_without_ls', 0), time_data.get('avg_with_ls', 0)]
        ax4.bar(categories, times)
        ax4.set_ylabel('平均耗时 (ms)')
        ax4.set_title('时间开销对比')

    # 5. 收敛曲线
    ax5 = axes[1, 1]
    if 'convergence_analysis' in validator.validation_results:
        convergence_data = validator.validation_results['convergence_analysis'].get('data', [])
        if convergence_data:
            rounds = [d['round'] for d in convergence_data]
            costs = [d['end_cost'] for d in convergence_data]
            ax5.plot(rounds, costs, 'g-o')
            ax5.set_xlabel('轮次')
            ax5.set_ylabel('成本')
            ax5.set_title('收敛行为')

    # 6. 综合评分雷达图
    ax6 = axes[1, 2]
    scores = validator._calculate_component_scores()
    if scores:
        categories = list(scores.keys())
        values = list(scores.values())

        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合
        angles += angles[:1]

        ax6 = plt.subplot(2, 3, 6, projection='polar')
        ax6.plot(angles, values, 'b-o')
        ax6.fill(angles, values, alpha=0.25)
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 100)
        ax6.set_title('综合性能评分')

    plt.tight_layout()
    plt.show()


def test_local_search_effectiveness(dynamic_optimizer):
    """
    主测试函数：全面测试局部搜索有效性
    """
    print("\n" + "=" * 80)
    print("🚀 开始局部搜索有效性测试")
    print("=" * 80)

    # 创建验证器
    validator = LocalSearchValidator(dynamic_optimizer)

    # 运行综合验证
    results = validator.run_comprehensive_validation(num_tests=5)

    # 可视化结果
    try:
        visualize_local_search_performance(validator)
    except Exception as e:
        print(f"\n⚠️ 可视化失败: {e}")

    # 提供改进建议
    print("\n" + "=" * 80)
    print("💡 改进建议:")
    print("=" * 80)

    if results.get('trigger_analysis', {}).get('trigger_rate', 0) < 50:
        print("1. ⚠️ 触发率偏低，建议:")
        print("   - 调整theta阈值，当前值可能过于严格")
        print("   - 考虑增加频率触发策略")

    if results.get('operator_performance'):
        low_performers = [
            op for op, stats in results['operator_performance'].items()
            if stats.get('success_rate', 0) < 20
        ]
        if low_performers:
            print(f"2. ⚠️ 低效算子: {low_performers}")
            print("   - 检查算子实现逻辑")
            print("   - 考虑约束条件是否过于严格")

    if results.get('improvement_tracking', {}).get('improvement_rate', 0) < 20:
        print("3. ⚠️ 改进率偏低，建议:")
        print("   - 增加局部搜索迭代次数")
        print("   - 尝试更激进的邻域操作")
        print("   - 考虑实现inter-route算子")

    if results.get('time_analysis', {}).get('overhead_rate', 0) > 200:
        print("4. ⚠️ 时间开销过大，建议:")
        print("   - 减少max_no_improve参数")
        print("   - 实现early stopping机制")
        print("   - 考虑概率性触发")

    print("\n测试完成！")
    return results


# 添加到LocalSearchValidator类中的辅助方法
def _calculate_component_scores(self):
    """计算各组件得分（用于雷达图）"""
    scores = {}

    if 'trigger_analysis' in self.validation_results:
        scores['触发机制'] = min(self.validation_results['trigger_analysis']['trigger_rate'], 100)

    if 'operator_performance' in self.validation_results:
        operator_stats = self.validation_results['operator_performance']
        if operator_stats:
            avg_success = np.mean([s.get('success_rate', 0) for s in operator_stats.values()])
            scores['算子性能'] = min(avg_success * 2, 100)

    if 'improvement_tracking' in self.validation_results:
        scores['改进效果'] = min(self.validation_results['improvement_tracking'].get('improvement_rate', 0) * 3, 100)

    if 'time_analysis' in self.validation_results:
        overhead = self.validation_results['time_analysis'].get('overhead_rate', 999)
        scores['时间效率'] = max(0, 100 - min(overhead, 100))

    if 'convergence_analysis' in self.validation_results:
        scores['收敛性'] = 100 if self.validation_results['convergence_analysis'].get('converged_at') else 0

    return scores


# 将这个方法添加到LocalSearchValidator类中
LocalSearchValidator._calculate_component_scores = _calculate_component_scores