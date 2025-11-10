import csv
import numpy as np
import random
import copy
import math
import traceback
from typing import List, Dict, Tuple, Optional

# ==================== 完整摧毁算子实现 ====================
class DestroyOperators:
    """约束版摧毁算子集合类 - 严格限制在车辆对内操作"""
    def __init__(self, dynamic_opt_instance):
        self.dyn_opt = dynamic_opt_instance
        # 算子权重和统计
        self.operator_weights = {
            'random': 1.0,
            'worst_distance': 1.0,
            'worst_time': 1.0,
            'drone_worst_energy': 1.0,
            'shaw': 1.0,
            'route': 1.0
        }
        # 算子性能统计
        self.operator_stats = {
            'random': {'calls': 0, 'improvements': 0},
            'worst_distance': {'calls': 0, 'improvements': 0},
            'worst_time': {'calls': 0, 'improvements': 0},
            'drone_worst_energy': {'calls': 0, 'improvements': 0},
            'shaw': {'calls': 0, 'improvements': 0},
            'route': {'calls': 0, 'improvements': 0}
        }

    def random_removal(self, truck_id, customer_id):
        """约束版随机移除算子 - 只在指定车辆对内删除客户"""
        print(f" 使用约束版随机移除算子（车辆对{truck_id}）")
        delete_list = []
        remain_list = []
        #  关键：只获取属于当前车辆对的客户
        vehicle_customers = self.dyn_opt.get_vehicle_customers(truck_id)
        if not vehicle_customers:
            print(f"    车辆对{truck_id}没有分配的客户")
            return []
        # 获取无人机节点信息（仅限当前车辆对）
        launch_node = []
        retrieval_node = []
        if self.dyn_opt.DRONE_Routes[truck_id].route:
            launch_node = [trip['launch_node'] for trip in self.dyn_opt.DRONE_Routes[truck_id].route]
            retrieval_node = [trip['retrieval_node'] for trip in self.dyn_opt.DRONE_Routes[truck_id].route]
        #  确定删除范围（仅限车辆对内）
        if customer_id == -1:  # 更新全部路径
            remain_list = list(vehicle_customers)
            print(f"    全路径更新模式：考虑车辆对{truck_id}的所有{len(remain_list)}个客户")
        else:
            #  验证触发客户属于当前车辆对
            if customer_id not in vehicle_customers:
                print(f"    触发客户{customer_id}不属于车辆对{truck_id}")
                return []
            # 收集该车辆对中当前客户之后的所有客户
            remain_list = self._get_remaining_customers_in_vehicle(truck_id, customer_id)
            print(f"    剩余路径更新模式：考虑客户{customer_id}之后的{len(remain_list)}个客户")
        #  过滤：确保所有候选客户都属于当前车辆对
        valid_remain_list = [c for c in remain_list if c in vehicle_customers]
        if len(valid_remain_list) != len(remain_list):
            invalid_count = len(remain_list) - len(valid_remain_list)
            print(f"    过滤了{invalid_count}个不属于车辆对{truck_id}的客户")
        remain_list = valid_remain_list
        if not remain_list:
            print(f"   ️ 车辆对{truck_id}没有可删除的客户")
            return []
        # 计算删除客户数量
        length = len(remain_list)
        min_remove = max(math.floor(length * self.dyn_opt.min_delete), 1)
        max_remove = max(math.floor(length * self.dyn_opt.max_delete), min_remove)
        remove_count = random.randint(min_remove, max_remove)
        print(f"    计划删除{remove_count}个客户 (范围: {min_remove}-{max_remove})")
        # 安全删除逻辑（仅在车辆对内）
        attempts = 0
        max_attempts = len(remain_list) * 2  # 避免无限循环
        while len(delete_list) < remove_count and len(remain_list) > 0 and attempts < max_attempts:
            attempts += 1
            selected_customer = random.choice(remain_list)
            if selected_customer in delete_list:
                continue
            # 执行约束版删除
            if self._safe_remove_customer(truck_id, selected_customer, delete_list, launch_node, retrieval_node):
                remain_list.remove(selected_customer)  # 从候选列表中移除
                print(f"      删除客户{selected_customer}")
            else:
                print(f"      客户{selected_customer}删除失败")
        # 最终验证：确保所有删除的客户都属于当前车辆对
        validated_delete_list = [c for c in delete_list if c in vehicle_customers]
        if len(validated_delete_list) != len(delete_list):
            invalid_deletes = [c for c in delete_list if c not in vehicle_customers]
            print(f"   ️ 发现{len(invalid_deletes)}个无效删除客户: {invalid_deletes}")
        print(f"    车辆对{truck_id}随机删除完成: {len(validated_delete_list)}个客户")
        return validated_delete_list

    def worst_distance_removal(self, truck_id, customer_id):
        """最差距离移除算子"""
        print(f"使用最差距离移除算子（车辆对{truck_id}）")
        delete_list = []
        remain_list = []
        # 获取车辆对客户
        vehicle_customers = self.dyn_opt.get_vehicle_customers(truck_id)
        if not vehicle_customers:
            return []
        # 获取无人机节点信息
        launch_node = []
        retrieval_node = []
        if self.dyn_opt.DRONE_Routes[truck_id].route:
            launch_node = [trip['launch_node'] for trip in self.dyn_opt.DRONE_Routes[truck_id].route]
            retrieval_node = [trip['retrieval_node'] for trip in self.dyn_opt.DRONE_Routes[truck_id].route]
        # 确定考虑范围（仅限车辆对内）
        if customer_id == -1:
            remain_list = list(vehicle_customers)
        else:
            if customer_id not in vehicle_customers:
                print(f"    客户{customer_id}不属于车辆对{truck_id}")
                return []
            remain_list = self._get_remaining_customers_in_vehicle(truck_id, customer_id)
        # 过滤确保都属于当前车辆对
        remain_list = [c for c in remain_list if c in vehicle_customers]
        if not remain_list:
            print(f"   ️ 车辆对{truck_id}没有可删除的客户")
            return []
        # 计算删除数量
        length = len(remain_list)
        min_remove = max(math.floor(length * self.dyn_opt.min_delete), 1)
        max_remove = max(math.floor(length * self.dyn_opt.max_delete), min_remove)
        remove_count = random.randint(min_remove, max_remove)
        # 计算每个客户的旅行成本（绕行距离）
        customer_costs = []
        for cust in remain_list:
            if cust in delete_list:
                continue
            travel_cost = self._calculate_detour_cost(truck_id, cust)
            customer_costs.append((cust, travel_cost))
        # 按旅行成本降序排序
        customer_costs.sort(key=lambda x: x[1], reverse=True)
        # 选择成本最高的客户进行删除
        candidates = [item[0] for item in customer_costs[:min(len(customer_costs), remove_count * 2)]]
        while len(delete_list) < remove_count and candidates:
            selected_customer = random.choice(candidates[:max(1, len(candidates) // 2)])
            candidates.remove(selected_customer)
            if selected_customer not in delete_list:
                self._safe_remove_customer(truck_id, selected_customer, delete_list, launch_node, retrieval_node)
        # 最终验证
        validated_delete_list = [c for c in delete_list if c in vehicle_customers]
        print(f"    车辆对{truck_id}最差距离删除完成: {len(validated_delete_list)}个高成本客户")
        return validated_delete_list

    def worst_time_removal(self, truck_id, customer_id):
        """最差时间移除算子"""
        print(f" 使用约束版最差时间移除算子（车辆对{truck_id}）")
        delete_list = []
        remain_list = []
        # 获取车辆对客户
        vehicle_customers = self.dyn_opt.get_vehicle_customers(truck_id)
        if not vehicle_customers:
            return []
        launch_node = []
        retrieval_node = []
        if self.dyn_opt.DRONE_Routes[truck_id].route:
            launch_node = [trip['launch_node'] for trip in self.dyn_opt.DRONE_Routes[truck_id].route]
            retrieval_node = [trip['retrieval_node'] for trip in self.dyn_opt.DRONE_Routes[truck_id].route]
        # 确定考虑范围
        if customer_id == -1:
            remain_list = list(vehicle_customers)
        else:
            if customer_id not in vehicle_customers:
                return []
            remain_list = self._get_remaining_customers_in_vehicle(truck_id, customer_id)
        remain_list = [c for c in remain_list if c in vehicle_customers]
        if not remain_list:
            return []
        # 计算删除数量
        length = len(remain_list)
        min_remove = max(math.floor(length * self.dyn_opt.min_delete), 1)
        max_remove = max(math.floor(length * self.dyn_opt.max_delete), min_remove)
        remove_count = random.randint(min_remove, max_remove)
        # 计算每个客户的时间窗偏差
        time_deviations = []
        for cust in remain_list:
            if cust in delete_list:
                continue
            customer_obj = self.dyn_opt.customers[cust - 1]
            service_start = customer_obj.service_begin if customer_obj.service_begin else customer_obj.start_time
            # 计算与理想服务时间的偏差
            ideal_time = (customer_obj.start_time + customer_obj.end_time) / 2
            deviation = abs(service_start - ideal_time) if service_start else 0
            time_deviations.append((cust, deviation))
        # 按时间偏差降序排序
        time_deviations.sort(key=lambda x: x[1], reverse=True)
        # 选择偏差最大的客户
        candidates = [item[0] for item in time_deviations[:min(len(time_deviations), remove_count * 2)]]
        while len(delete_list) < remove_count and candidates:
            selected_customer = random.choice(candidates[:max(1, len(candidates) // 2)])
            candidates.remove(selected_customer)
            if selected_customer not in delete_list:
                self._safe_remove_customer(truck_id, selected_customer, delete_list, launch_node, retrieval_node)
        validated_delete_list = [c for c in delete_list if c in vehicle_customers]
        print(f"    车辆对{truck_id}最差时间删除完成: {len(validated_delete_list)}个时间偏差大的客户")
        return validated_delete_list

    def drone_worst_energy_removal(self, truck_id, customer_id):
        """无人机最差能耗移除算子"""
        print(f" 使用约束版无人机最差能耗移除算子（车辆对{truck_id}）")
        # 检查该车辆对是否有无人机路径
        if not self.dyn_opt.DRONE_Routes[truck_id].route:
            print(f"    车辆对{truck_id}没有无人机路径，回退到随机移除")
            return self.random_removal(truck_id, customer_id)
        delete_list = []
        remain_list = []
        # 获取车辆对客户
        vehicle_customers = self.dyn_opt.get_vehicle_customers(truck_id)
        if not vehicle_customers:
            return []
        launch_node = [trip['launch_node'] for trip in self.dyn_opt.DRONE_Routes[truck_id].route]
        retrieval_node = [trip['retrieval_node'] for trip in self.dyn_opt.DRONE_Routes[truck_id].route]
        # 确定范围
        if customer_id == -1:
            remain_list = list(vehicle_customers)
        else:
            if customer_id not in vehicle_customers:
                return []
            remain_list = self._get_remaining_customers_in_vehicle(truck_id, customer_id)
        # 只考虑真正由无人机服务的客户（在当前车辆对内）
        drone_customers = []
        for customer in remain_list:
            if (customer in vehicle_customers and  # 确保属于当前车辆对
                    customer - 1 < len(self.dyn_opt.customers) and
                    hasattr(self.dyn_opt.customers[customer - 1], 'service_by') and
                    self.dyn_opt.customers[customer - 1].service_by and
                    self.dyn_opt.customers[customer - 1].service_by[0] == "de" and
                    self.dyn_opt.customers[customer - 1].service_by[1] == truck_id):  # 确保由当前车辆对的无人机服务
                # 再次确认客户在当前车辆对的无人机路径中
                for trip in self.dyn_opt.DRONE_Routes[truck_id].route:
                    if customer in trip['path']:
                        drone_customers.append(customer)
                        break
        if not drone_customers:
            print(f"   ️ 车辆对{truck_id}没有无人机服务的客户，回退到随机移除")
            return self.random_removal(truck_id, customer_id)
        # 计算删除数量
        length = len(drone_customers)
        min_remove = max(math.floor(length * self.dyn_opt.min_delete), 1)
        max_remove = max(math.floor(length * self.dyn_opt.max_delete), min_remove)
        remove_count = random.randint(min_remove, max_remove)
        # 计算每个无人机服务客户的等待能耗
        energy_costs = []
        for cust in drone_customers:
            if cust in delete_list:
                continue
            waiting_energy = self._calculate_drone_waiting_energy(truck_id, cust)
            energy_costs.append((cust, waiting_energy))
        if not energy_costs:
            return self.random_removal(truck_id, customer_id)
        # 按等待能耗降序排序
        energy_costs.sort(key=lambda x: x[1], reverse=True)
        # 选择能耗最高的客户
        candidates = [item[0] for item in energy_costs[:min(len(energy_costs), remove_count * 2)]]
        while len(delete_list) < remove_count and candidates:
            selected_customer = random.choice(candidates[:max(1, len(candidates) // 2)])
            candidates.remove(selected_customer)
            if selected_customer not in delete_list:
                self._safe_remove_customer(truck_id, selected_customer, delete_list, launch_node, retrieval_node)
        validated_delete_list = [c for c in delete_list if c in vehicle_customers]
        print(f"    车辆对{truck_id}最差能耗删除完成: {len(validated_delete_list)}个高能耗客户")
        return validated_delete_list

    def shaw_removal(self, truck_id, customer_id):
        """Shaw相似性移除算子"""
        print(f" 使用约束版Shaw相似性移除算子（车辆对{truck_id}）")
        delete_list = []
        remain_list = []
        # 获取车辆对客户
        vehicle_customers = self.dyn_opt.get_vehicle_customers(truck_id)
        if not vehicle_customers:
            return []
        launch_node = []
        retrieval_node = []
        if self.dyn_opt.DRONE_Routes[truck_id].route:
            launch_node = [trip['launch_node'] for trip in self.dyn_opt.DRONE_Routes[truck_id].route]
            retrieval_node = [trip['retrieval_node'] for trip in self.dyn_opt.DRONE_Routes[truck_id].route]
        # 确定范围
        if customer_id == -1:
            remain_list = list(vehicle_customers)
        else:
            if customer_id not in vehicle_customers:
                return []
            remain_list = self._get_remaining_customers_in_vehicle(truck_id, customer_id)
        remain_list = [c for c in remain_list if c in vehicle_customers]
        if not remain_list:
            return []
        # 计算删除数量
        length = len(remain_list)
        min_remove = max(math.floor(length * self.dyn_opt.min_delete), 1)
        max_remove = max(math.floor(length * self.dyn_opt.max_delete), min_remove)
        remove_count = random.randint(min_remove, max_remove)
        # 随机选择种子客户
        seed_customer = random.choice(remain_list)
        # 确保种子客户被删除
        if self._safe_remove_customer(truck_id, seed_customer, delete_list, launch_node, retrieval_node):
            remain_list.remove(seed_customer)
        # 计算相似性权重
        phi_1, phi_2, phi_3 = 0.4, 0.3, 0.3  # 距离、时间、服务类型权重
        # 迭代添加最相似的客户
        while len(delete_list) < remove_count and len(remain_list) > 0:
            if not remain_list:
                break
            similarities = []
            for candidate in remain_list:
                if candidate in delete_list:
                    continue
                try:
                    similarity = self._calculate_shaw_similarity(seed_customer, candidate, phi_1, phi_2, phi_3)
                    similarities.append((candidate, similarity))
                except (AttributeError, IndexError, KeyError):
                    similarities.append((candidate, random.random()))
            if not similarities:
                break
            # 按相似性降序排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            # 选择最相似的客户
            next_customer = similarities[0][0]
            if self._safe_remove_customer(truck_id, next_customer, delete_list, launch_node, retrieval_node):
                remain_list.remove(next_customer)
        validated_delete_list = [c for c in delete_list if c in vehicle_customers]
        print(f"    车辆对{truck_id}Shaw相似性删除完成: {len(validated_delete_list)}个相似客户")
        return validated_delete_list

    def route_removal(self, truck_id, customer_id):
        """路径移除算子"""
        print(f"️ 使用约束版路径移除算子（车辆对{truck_id}）")
        delete_list = []
        # 只在当前车辆对内寻找最小路径
        min_customers = float('inf')
        target_route_type = None  # 'truck' 或 'drone'
        target_route_index = None
        # 检查当前车辆对的卡车路径
        truck_customers = len(self.dyn_opt.TRUCK_Routes[truck_id].Troute) - 2  # 排除起终点
        if truck_customers < min_customers and truck_customers > 0:
            min_customers = truck_customers
            target_route_type = 'truck'
        # 检查当前车辆对的无人机路径
        if self.dyn_opt.DRONE_Routes[truck_id].route:
            for idx, trip in enumerate(self.dyn_opt.DRONE_Routes[truck_id].route):
                drone_customers = len(trip['path']) - 2  # 排除起终点
                if drone_customers < min_customers and drone_customers > 0:
                    min_customers = drone_customers
                    target_route_type = 'drone'
                    target_route_index = idx
        # 在当前车辆对内执行路径移除
        vehicle_customers = self.dyn_opt.get_vehicle_customers(truck_id)
        if target_route_type == 'truck':
            # 移除卡车路径中的客户
            for i in range(1, len(self.dyn_opt.TRUCK_Routes[truck_id].Troute) - 1):
                customer = self.dyn_opt.TRUCK_Routes[truck_id].Troute[i]
                if customer in vehicle_customers:  # 确保属于当前车辆对
                    delete_list.append(customer)
            # 清空卡车路径（保留起终点）
            self.dyn_opt.TRUCK_Routes[truck_id].Troute = [
                self.dyn_opt.TRUCK_Routes[truck_id].Troute[0],
                self.dyn_opt.TRUCK_Routes[truck_id].Troute[-1]
            ]
            print(f"    删除车辆对{truck_id}的整个卡车路径")
        elif target_route_type == 'drone' and target_route_index is not None:
            # 移除无人机路径
            trip = self.dyn_opt.DRONE_Routes[truck_id].route[target_route_index]
            for i in range(1, len(trip['path']) - 1):
                customer = trip['path'][i]
                if customer in vehicle_customers:  # 确保属于当前车辆对
                    delete_list.append(customer)
            # 删除该无人机行程
            del self.dyn_opt.DRONE_Routes[truck_id].route[target_route_index]
            print(f"    删除车辆对{truck_id}的无人机路径{target_route_index}")
        # 如果没有找到合适路径，回退到随机移除
        if not delete_list:
            print(f"    车辆对{truck_id}没有合适的路径删除，回退到随机移除")
            return self.random_removal(truck_id, customer_id)
        # 最终验证
        validated_delete_list = [c for c in delete_list if c in vehicle_customers]
        print(f"    车辆对{truck_id}路径删除完成: {len(validated_delete_list)}个客户")
        return validated_delete_list

    def select_destroy_operator(self):
        """基于轮盘赌选择摧毁算子"""
        operators = list(self.operator_weights.keys())
        weights = list(self.operator_weights.values())
        # 归一化权重
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        # 轮盘赌选择
        selected = np.random.choice(operators, p=probabilities)
        return selected

    def update_operator_performance(self, operator_name, improved):
        """更新算子性能统计"""
        self.operator_stats[operator_name]['calls'] += 1
        if improved:
            self.operator_stats[operator_name]['improvements'] += 1
        # 每50次调用更新一次权重
        if self.operator_stats[operator_name]['calls'] % 50 == 0:
            self._update_operator_weights()

    def _update_operator_weights(self):
        """基于性能统计更新算子权重"""
        for op_name, stats in self.operator_stats.items():
            if stats['calls'] > 0:
                success_rate = stats['improvements'] / stats['calls']
                # 权重调整：成功率越高权重越大
                self.operator_weights[op_name] = 0.1 + success_rate * 0.9

    # 辅助方法
    def _get_remaining_customers_in_vehicle(self, truck_id: int, current_customer_id: int) -> list:
        """
        获取指定车辆对中指定客户之后的所有客户
        """
        remaining_customers = []
        vehicle_customers = self.dyn_opt.get_vehicle_customers(truck_id)
        # 从卡车路径中收集
        truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
        if current_customer_id in truck_route:
            current_idx = truck_route.index(current_customer_id)
            # 只收集后续的客户，且确保属于当前车辆对
            for i in range(current_idx + 1, len(truck_route) - 1):  # 排除终点
                customer = truck_route[i]
                if customer in vehicle_customers:
                    remaining_customers.append(customer)
        # 从无人机路径中收集（只考虑当前车辆对的无人机）
        for trip in self.dyn_opt.DRONE_Routes[truck_id].route:
            if current_customer_id in trip['path']:
                retrieval_node = trip['retrieval_node']
                if retrieval_node in truck_route:
                    retrieval_idx = truck_route.index(retrieval_node)
                    # 收集回收节点之后的卡车路径客户
                    for i in range(retrieval_idx + 1, len(truck_route) - 1):
                        customer = truck_route[i]
                        if customer in vehicle_customers:
                            remaining_customers.append(customer)
                    # 收集该回收节点之后的无人机任务中的客户
                    for other_trip in self.dyn_opt.DRONE_Routes[truck_id].route:
                        if (other_trip['launch_node'] in truck_route[retrieval_idx:] and
                                other_trip['launch_node'] in vehicle_customers):
                            for drone_customer in other_trip['path'][1:-1]:
                                if drone_customer in vehicle_customers:
                                    remaining_customers.append(drone_customer)
        return list(set(remaining_customers))  # 去重

    def _safe_remove_customer(self, truck_id, selected_customer, delete_list, launch_node, retrieval_node):
        """
        安全地移除客户，确保只在指定车辆对内操作
        """
        # 首先验证客户属于当前车辆对
        if not self.dyn_opt.validate_customer_assignment(truck_id, selected_customer):
            print(f"      客户{selected_customer}不属于车辆对{truck_id}，跳过删除")
            return False
        if selected_customer in delete_list:
            return False
        customer_actually_removed = False
        try:
            # 1. 从指定车辆对的卡车路径中删除
            if selected_customer in self.dyn_opt.TRUCK_Routes[truck_id].Troute:
                self.dyn_opt.TRUCK_Routes[truck_id].Troute.remove(selected_customer)
                delete_list.append(selected_customer)
                customer_actually_removed = True
                # 处理起飞/回收节点的依赖关系（仅限当前车辆对）
                if selected_customer in launch_node or selected_customer in retrieval_node:
                    trips_to_remove = []
                    for i, trip in enumerate(self.dyn_opt.DRONE_Routes[truck_id].route):
                        if (trip['launch_node'] == selected_customer or
                                trip['retrieval_node'] == selected_customer):
                            # 将该trip中的所有客户加入删除列表（验证属于当前车辆对）
                            for drone_customer in trip['path'][1:-1]:  # 排除起点终点
                                if (drone_customer not in delete_list and
                                        self.dyn_opt.validate_customer_assignment(truck_id, drone_customer)):
                                    delete_list.append(drone_customer)
                            trips_to_remove.append(i)
                    # 逆序删除trip以避免索引问题
                    for i in reversed(trips_to_remove):
                        del self.dyn_opt.DRONE_Routes[truck_id].route[i]
            # 2. 从指定车辆对的无人机路径中删除
            else:
                trips_to_update = []
                for trip_idx, trip in enumerate(self.dyn_opt.DRONE_Routes[truck_id].route):
                    if selected_customer in trip['path']:
                        trip['path'].remove(selected_customer)
                        delete_list.append(selected_customer)
                        customer_actually_removed = True
                        # 如果路径只剩起终点，标记删除整个trip
                        if len(trip['path']) <= 2:
                            trips_to_update.append(trip_idx)
                        else:
                            # 重新计算载重（只考虑当前车辆对的客户）
                            trip['current_load'] = sum(
                                self.dyn_opt.customers[c - 1].demand
                                for c in trip['path'][1:-1]
                                if (c <= len(self.dyn_opt.customers) and
                                    self.dyn_opt.customers[c - 1].demand > 0 and
                                    self.dyn_opt.validate_customer_assignment(truck_id, c))
                            )
                            trip['current_load_delivery'] = trip['current_load']
                            trip['initial_load'] = trip['current_load']
                            trip['initial_load_delivery'] = trip['current_load']
                        break
                # 删除空的trips
                for trip_idx in reversed(trips_to_update):
                    del self.dyn_opt.DRONE_Routes[truck_id].route[trip_idx]
            return customer_actually_removed
        except (ValueError, IndexError, KeyError) as e:
            print(f"     ️ 移除客户{selected_customer}时出现错误: {e}")
            # 确保客户至少被从delete_list中移除，避免无限循环
            if selected_customer not in delete_list:
                delete_list.append(selected_customer)
            return False

    def _calculate_detour_cost(self, truck_id, customer):
        """计算指定车辆对内客户的绕行成本"""
        try:
            # 只在指定车辆对的卡车路径中查找
            if customer in self.dyn_opt.TRUCK_Routes[truck_id].Troute:
                route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
                idx = route.index(customer)
                if idx == 0 or idx == len(route) - 1:
                    return 0
                prev_node = route[idx - 1]
                next_node = route[idx + 1]
                # 计算绕行距离
                original_dist = self.dyn_opt.ALLdistanceTmatrix[prev_node][next_node]
                detour_dist = (self.dyn_opt.ALLdistanceTmatrix[prev_node][customer] +
                               self.dyn_opt.ALLdistanceTmatrix[customer][next_node])
                return detour_dist - original_dist
            return 1.0
        except (IndexError, KeyError, ValueError):
            return 1.0

    def _calculate_drone_waiting_energy(self, truck_id, customer):
        """计算指定车辆对内无人机在客户处的等待能耗"""
        try:
            customer_obj = self.dyn_opt.customers[customer - 1]
            # 验证客户属于指定车辆对且由无人机服务
            if (self.dyn_opt.validate_customer_assignment(truck_id, customer) and
                    customer_obj.service_by[0] == "de" and
                    customer_obj.service_by[1] == truck_id and
                    hasattr(customer_obj, 'arrive_drone') and
                    hasattr(customer_obj, 'start_time')):
                arrive_time = customer_obj.arrive_drone
                start_time = customer_obj.start_time
                if arrive_time and start_time and arrive_time < start_time:
                    wait_time = start_time - arrive_time
                    # 简化的等待能耗计算
                    return wait_time * self.dyn_opt.energy_hover * self.dyn_opt.drone_weight
            return 0.0
        except (AttributeError, IndexError, KeyError):
            return 0.0

    def _calculate_shaw_similarity(self, customer1, customer2, phi_1, phi_2, phi_3):
        """计算同一车辆对内两个客户之间的Shaw相似性"""
        try:
            cust1_obj = self.dyn_opt.customers[customer1 - 1]
            cust2_obj = self.dyn_opt.customers[customer2 - 1]
            # 距离相似性
            dist_similarity = self.dyn_opt.ALLdistanceTmatrix[customer1][customer2]
            # 时间窗相似性
            time_similarity = abs(cust1_obj.start_time - cust2_obj.start_time)
            # 服务类型相似性（同一车辆对服务为1，否则为0）
            service_similarity = 1 if (cust1_obj.service_by[1] == cust2_obj.service_by[1]) else 0
            # 综合相似性得分（越大越相似）
            similarity = phi_1 * dist_similarity + phi_2 * time_similarity + phi_3 * service_similarity
            return similarity
        except (AttributeError, IndexError, KeyError):
            return 0.0

# ==================== 完整修复算子实现 ====================
class RepairOperators:
    """修复算子集合类 - 严格限制在车辆对内修复"""
    def __init__(self, dynamic_opt_instance):
        self.dyn_opt = dynamic_opt_instance
        # 算子权重和统计
        self.operator_weights = {
            'random_order': 1.0,
            'cheapest_distance': 1.0,
            'regret_distance': 1.0,
            'greedy_feasibility': 1.0,
            'drone_priority': 1.0,
            'drone_newroute': 1.0
        }
        # 算子性能统计
        self.operator_stats = {
            'random_order': {'calls': 0, 'improvements': 0, 'success_rate': 0.0},
            'cheapest_distance': {'calls': 0, 'improvements': 0, 'success_rate': 0.0},
            'regret_distance': {'calls': 0, 'improvements': 0, 'success_rate': 0.0},
            'greedy_feasibility': {'calls': 0, 'improvements': 0, 'success_rate': 0.0},
            'drone_priority': {'calls': 0, 'improvements': 0, 'success_rate': 0.0},
            'drone_newroute': {'calls': 0, 'improvements': 0, 'success_rate': 0.0}
        }
        # 插入策略选择参数
        self.insertion_attempts_limit = 50
        self.feasibility_check_enabled = True

    def repair_solution(self, truck_id: int, delete_list: List[int]) -> bool:
        """
        严格限制在指定车辆对内修复
        """
        if not delete_list:
            return True
        delete_list = self._enforce_package_constraints(truck_id, delete_list)
        if not delete_list:
            print(f"    经约束过滤后，车辆对{truck_id}没有可修复的客户")
            return True
        # 关键验证：确保所有待修复客户都属于指定车辆对
        vehicle_customers = self.dyn_opt.get_vehicle_customers(truck_id)
        invalid_customers = [c for c in delete_list if c not in vehicle_customers]
        if invalid_customers:
            print(f" 发现不属于车辆对{truck_id}的客户: {invalid_customers}")
            for invalid_customer in invalid_customers:
                actual_vehicle = self.dyn_opt.get_customer_vehicle(invalid_customer)
                print(f"   客户{invalid_customer}实际属于车辆对: {actual_vehicle}")
            return False
        # 选择修复算子
        selected_operator = self.select_repair_operator()
        print(f"🔧 选择修复算子: {selected_operator} (仅限车辆对{truck_id}内)")
        # 记录修复前的成本
        cost_before = self.dyn_opt.cost()
        # 执行约束版修复算子
        success = False
        try:
            if selected_operator == 'random_order':
                success = self.random_order_insertion(truck_id, delete_list)
            elif selected_operator == 'cheapest_distance':
                success = self.cheapest_distance_insertion(truck_id, delete_list)
            elif selected_operator == 'regret_distance':
                success = self.regret_distance_insertion(truck_id, delete_list)
            elif selected_operator == 'greedy_feasibility':
                success = self.greedy_feasibility_insertion(truck_id, delete_list)
            elif selected_operator == 'drone_priority':
                success = self.drone_priority_insertion(truck_id, delete_list)
            elif selected_operator == 'drone_newroute':
                success = self.drone_newroute_insertion(truck_id, delete_list)
            else:
                success = self.greedy_feasibility_insertion(truck_id, delete_list)
        except Exception as e:
            print(f"    修复算子执行失败: {e}")
            success = self.emergency_repair(truck_id, delete_list)
        if success:
            print(f"    执行车辆对{truck_id}的可行性检查...")
            feasibility_success = self.dyn_opt.feasibility_repair_ops.check_and_repair_feasibility(truck_id)
            if not feasibility_success:
                print(f"    ⚠️ 车辆对{truck_id}可行性修复失败")
                # 可以选择是否将此视为整体失败
                # success = False  # 取消注释以严格要求可行性
            else:
                print(f"    ✅ 车辆对{truck_id}所有约束满足")
        # 评估修复效果并更新统计
        if success:
            cost_after = self.dyn_opt.cost()
            improved = cost_after < cost_before
            self.update_operator_performance(selected_operator, improved)
            print(f"    车辆对{truck_id}修复成功，成本变化: {cost_before:.2f} → {cost_after:.2f}")
        else:
            print(f"    车辆对{truck_id}修复失败，使用应急策略")
            success = self.emergency_repair(truck_id, delete_list)
            self.update_operator_performance(selected_operator, False)
        return success

    def random_order_insertion(self, truck_id: int, delete_list: List[int]) -> bool:
        """
        随机顺序插入：只能插入到指定车辆对中
        """
        print(f"   🎲 执行随机顺序插入（车辆对{truck_id}）...")
        shuffled_customers = delete_list.copy()
        random.shuffle(shuffled_customers)
        inserted_count = 0
        for customer_id in shuffled_customers:
            #  关键：只尝试插入到指定truck_id的路径中
            if self._insert_customer_to_specific_vehicle(truck_id, customer_id):
                inserted_count += 1
                print(f"       客户{customer_id}插入到车辆对{truck_id}")
            else:
                print(f"       客户{customer_id}插入失败")
        success_rate = (inserted_count / len(delete_list)) * 100 if delete_list else 100
        print(f"    车辆对{truck_id}随机插入完成: {inserted_count}/{len(delete_list)} ({success_rate:.1f}%)")
        return inserted_count == len(delete_list)

    def cheapest_distance_insertion(self, truck_id: int, delete_list: List[int]) -> bool:
        """
        最便宜距离插入：每次选择在指定车辆对内插入成本最低的客户和位置
        """
        print(f"    执行最便宜距离插入（车辆对{truck_id}）...")
        remaining_customers = delete_list.copy()
        inserted_count = 0
        while remaining_customers:
            best_insertion = None
            best_cost = float('inf')
            #  为每个剩余客户在指定车辆对内找到最便宜的插入位置
            for customer_id in remaining_customers:
                insertion_options = self._get_vehicle_insertion_options(truck_id, customer_id)
                for option in insertion_options:
                    if option['cost'] < best_cost:
                        best_cost = option['cost']
                        best_insertion = {
                            'customer_id': customer_id,
                            'option': option
                        }
            # 执行最佳插入
            if best_insertion and self._execute_insertion(truck_id, best_insertion):
                remaining_customers.remove(best_insertion['customer_id'])
                inserted_count += 1
                print(f"       客户{best_insertion['customer_id']}插入，成本增加: {best_cost:.2f}")
            else:
                # 应急处理
                if remaining_customers:
                    customer_id = remaining_customers.pop(0)
                    if self._insert_customer_to_specific_vehicle(truck_id, customer_id):
                        inserted_count += 1
                    else:
                        print(f"       客户{customer_id}无法插入车辆对{truck_id}")
                        break
        success_rate = (inserted_count / len(delete_list)) * 100 if delete_list else 100
        print(f"    车辆对{truck_id}最便宜插入完成: {inserted_count}/{len(delete_list)} ({success_rate:.1f}%)")
        return inserted_count == len(delete_list)

    def regret_distance_insertion(self, truck_id: int, delete_list: List[int]) -> bool:
        """
        约束版后悔距离插入：在指定车辆对内优先插入"后悔值"最大的客户
        """
        print(f"执行后悔距离插入（车辆对{truck_id}）...")
        remaining_customers = delete_list.copy()
        inserted_count = 0
        while remaining_customers:
            best_insertion = None
            max_regret = -1
            # 为每个客户在指定车辆对内计算后悔值
            for customer_id in remaining_customers:
                insertion_options = self._get_vehicle_insertion_options(truck_id, customer_id)
                if len(insertion_options) >= 2:
                    # 按成本排序
                    insertion_options.sort(key=lambda x: x['cost'])
                    # 计算后悔值：次优成本 - 最优成本
                    regret = insertion_options[1]['cost'] - insertion_options[0]['cost']
                    if regret > max_regret:
                        max_regret = regret
                        best_insertion = {
                            'customer_id': customer_id,
                            'option': insertion_options[0],  # 选择最优位置
                            'regret': regret
                        }
                elif len(insertion_options) == 1:
                    # 只有一个选择时，后悔值设为无穷大
                    if max_regret < float('inf'):
                        max_regret = float('inf')
                        best_insertion = {
                            'customer_id': customer_id,
                            'option': insertion_options[0],
                            'regret': float('inf')
                        }
            # 执行最大后悔值的插入
            if best_insertion and self._execute_insertion(truck_id, best_insertion):
                remaining_customers.remove(best_insertion['customer_id'])
                inserted_count += 1
                print(f"       客户{best_insertion['customer_id']}插入，后悔值: {best_insertion['regret']:.2f}")
            else:
                # 应急处理
                if remaining_customers:
                    customer_id = remaining_customers.pop(0)
                    if self._insert_customer_to_specific_vehicle(truck_id, customer_id):
                        inserted_count += 1
                    else:
                        print(f"       客户{customer_id}无法插入车辆对{truck_id}")
                        break
        success_rate = (inserted_count / len(delete_list)) * 100 if delete_list else 100
        print(f"    车辆对{truck_id}后悔插入完成: {inserted_count}/{len(delete_list)} ({success_rate:.1f}%)")
        return inserted_count == len(delete_list)

    def greedy_feasibility_insertion(self, truck_id: int, delete_list: List[int]) -> bool:
        """
        贪婪可行性插入：在指定车辆对内快速插入到第一个可行位置
        """
        print(f"   ⚡ 执行贪婪可行性插入（车辆对{truck_id}）...")
        # 按需求大小排序，优先处理大需求客户
        sorted_customers = sorted(delete_list,
                                  key=lambda c: abs(self.dyn_opt.customers[c - 1].demand),
                                  reverse=True)
        inserted_count = 0
        for customer_id in sorted_customers:
            if self._insert_customer_to_specific_vehicle(truck_id, customer_id):
                inserted_count += 1
                print(f"       客户{customer_id}快速插入车辆对{truck_id}")
            else:
                print(f"       客户{customer_id}插入车辆对{truck_id}失败")
        success_rate = (inserted_count / len(delete_list)) * 100 if delete_list else 100
        print(f"   📊 车辆对{truck_id}贪婪插入完成: {inserted_count}/{len(delete_list)} ({success_rate:.1f}%)")
        return inserted_count == len(delete_list)

    def drone_priority_insertion(self, truck_id: int, delete_list: List[int]) -> bool:
        """
        无人机优先插入：在指定车辆对内优先尝试将客户分配给无人机服务
        """
        print(f"执行无人机优先插入（车辆对{truck_id}）...")
        drone_customers = []
        truck_customers = []
        # 分类客户
        for customer_id in delete_list:
            customer = self.dyn_opt.customers[customer_id - 1]
            if (customer.drone_eligible == 1 and
                    abs(customer.demand) <= self.dyn_opt.drone_max_capacity):
                drone_customers.append(customer_id)
            else:
                truck_customers.append(customer_id)
        inserted_count = 0
        # 1. 优先处理无人机客户（仅限指定车辆对）
        print(f" 处理{len(drone_customers)}个无人机适用客户...")
        for customer_id in drone_customers:
            if self._try_insert_to_vehicle_drone(truck_id, customer_id):
                inserted_count += 1
                print(f"          客户{customer_id}插入车辆对{truck_id}无人机路径")
            elif self._try_insert_to_vehicle_truck(truck_id, customer_id):
                inserted_count += 1
                print(f"          客户{customer_id}回退到车辆对{truck_id}卡车路径")
            else:
                print(f"          客户{customer_id}插入车辆对{truck_id}失败")
        # 2. 处理卡车客户（仅限指定车辆对）
        print(f"       处理{len(truck_customers)}个卡车专用客户...")
        for customer_id in truck_customers:
            if self._try_insert_to_vehicle_truck(truck_id, customer_id):
                inserted_count += 1
                print(f"          客户{customer_id}插入车辆对{truck_id}卡车路径")
            else:
                print(f"          客户{customer_id}插入车辆对{truck_id}失败")
        success_rate = (inserted_count / len(delete_list)) * 100 if delete_list else 100
        print(f"   📊 车辆对{truck_id}无人机优先插入完成: {inserted_count}/{len(delete_list)} ({success_rate:.1f}%)")
        return inserted_count == len(delete_list)

    def drone_newroute_insertion(self, truck_id: int, delete_list: List[int]) -> bool:
        """
        无人机新路径插入：为指定车辆对的无人机客户创建新的飞行路径
        """
        print(f" 执行无人机新路径插入（车辆对{truck_id}）...")
        drone_customers = []
        other_customers = []
        # 分类客户
        for customer_id in delete_list:
            customer = self.dyn_opt.customers[customer_id - 1]
            if (customer.drone_eligible == 1 and
                    abs(customer.demand) <= self.dyn_opt.drone_max_capacity):
                drone_customers.append(customer_id)
            else:
                other_customers.append(customer_id)
        inserted_count = 0
        # 1. 尝试为无人机客户在指定车辆对内创建新路径
        print(f"       尝试为{len(drone_customers)}个客户创建车辆对{truck_id}的新无人机路径...")
        remaining_drone_customers = drone_customers.copy()
        while remaining_drone_customers:
            #关键：只在指定车辆对内创建无人机路径
            new_route_customers = self._create_new_drone_route(truck_id, remaining_drone_customers)
            if new_route_customers:
                inserted_count += len(new_route_customers)
                for cust_id in new_route_customers:
                    remaining_drone_customers.remove(cust_id)
                print(f"         为车辆对{truck_id}创建新路径，包含客户: {new_route_customers}")
            else:
                # 如果无法创建新路径，回退到现有路径插入
                customer_id = remaining_drone_customers.pop(0)
                if self._try_insert_to_vehicle_drone(truck_id, customer_id):
                    inserted_count += 1
                    print(f"          客户{customer_id}插入车辆对{truck_id}现有无人机路径")
                elif self._insert_customer_to_specific_vehicle(truck_id, customer_id):
                    inserted_count += 1
                    print(f"          客户{customer_id}回退到车辆对{truck_id}卡车路径")
                else:
                    print(f"          客户{customer_id}插入车辆对{truck_id}失败")
                    break
        # 2. 处理其他客户（仅限指定车辆对）
        print(f"       处理{len(other_customers)}个非无人机客户...")
        for customer_id in other_customers:
            if self._insert_customer_to_specific_vehicle(truck_id, customer_id):
                inserted_count += 1
            else:
                print(f"          客户{customer_id}插入车辆对{truck_id}失败")
        success_rate = (inserted_count / len(delete_list)) * 100 if delete_list else 100
        print(f"    车辆对{truck_id}新路径插入完成: {inserted_count}/{len(delete_list)} ({success_rate:.1f}%)")
        return inserted_count == len(delete_list)

    def select_repair_operator(self):
        """基于轮盘赌选择修复算子"""
        operators = list(self.operator_weights.keys())
        weights = list(self.operator_weights.values())
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(operators)
        probabilities = [w / total_weight for w in weights]
        selected = np.random.choice(operators, p=probabilities)
        return selected

    def update_operator_performance(self, operator_name: str, improved: bool):
        """更新算子性能统计"""
        self.operator_stats[operator_name]['calls'] += 1
        if improved:
            self.operator_stats[operator_name]['improvements'] += 1
        # 更新成功率
        calls = self.operator_stats[operator_name]['calls']
        improvements = self.operator_stats[operator_name]['improvements']
        self.operator_stats[operator_name]['success_rate'] = improvements / calls if calls > 0 else 0.0
        # 每50次调用更新一次权重
        if calls % 50 == 0:
            self._update_operator_weights()

    def _update_operator_weights(self):
        """基于性能统计更新算子权重"""
        for op_name, stats in self.operator_stats.items():
            if stats['calls'] > 10:
                success_rate = stats['success_rate']
                base_weight = 0.1
                performance_weight = success_rate * 0.8
                exploration_weight = 0.1
                self.operator_weights[op_name] = base_weight + performance_weight + exploration_weight

    # 辅助方法
    def _insert_customer_to_specific_vehicle(self, truck_id: int, customer_id: int) -> bool:
        """
        将客户插入到指定车辆对中（严格禁止跨车辆对）
        """
        # 首先验证客户属于指定车辆对
        if not self.dyn_opt.validate_customer_assignment(truck_id, customer_id):
            print(f"       客户{customer_id}不属于车辆对{truck_id}")
            return False
        # 1. 尝试插入到指定卡车路径
        if self._try_insert_to_vehicle_truck(truck_id, customer_id):
            return True
        # 2. 尝试插入到指定无人机路径
        if self._try_insert_to_vehicle_drone(truck_id, customer_id):
            return True
        return False

    def _try_insert_to_vehicle_truck(self, truck_id: int, customer_id: int) -> bool:
        """尝试将客户插入到指定车辆对的卡车路径中"""
        truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
        # 尝试每个位置
        for pos in range(1, len(truck_route)):
            if self._is_truck_insertion_feasible(truck_id, customer_id, pos):
                return self._insert_customer_to_truck_at_position(truck_id, customer_id, pos)
        return False

    def _try_insert_to_vehicle_drone(self, truck_id: int, customer_id: int) -> bool:
        """尝试将客户插入到指定车辆对的无人机路径中"""
        customer = self.dyn_opt.customers[customer_id - 1]
        # 检查无人机适用性
        if (customer.drone_eligible != 1 or
                abs(customer.demand) > self.dyn_opt.drone_max_capacity):
            return False
        # 尝试插入现有无人机路径
        for trip_idx, trip in enumerate(self.dyn_opt.DRONE_Routes[truck_id].route):
            path = trip['path']
            for pos in range(1, len(path)):
                if self._is_drone_insertion_feasible(truck_id, trip_idx, customer_id, pos):
                    return self._insert_customer_to_drone_at_position(
                        truck_id, customer_id, trip_idx, pos)
        return False

    def _get_vehicle_insertion_options(self, truck_id: int, customer_id: int) -> List[Dict]:
        """
        获取客户在指定车辆对内的所有可行插入选项
        """
        options = []
        # 1. 指定车辆对的卡车路径插入选项
        truck_options = self._get_truck_insertion_options(truck_id, customer_id)
        options.extend(truck_options)
        # 2. 指定车辆对的无人机路径插入选项
        drone_options = self._get_drone_insertion_options(truck_id, customer_id)
        options.extend(drone_options)
        return options

    def _get_truck_insertion_options(self, truck_id: int, customer_id: int) -> List[Dict]:
        """获取指定车辆对卡车路径的插入选项"""
        options = []
        truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
        customer = self.dyn_opt.customers[customer_id - 1]
        # 尝试每个可能的插入位置
        for pos in range(1, len(truck_route)):
            if self._is_truck_insertion_feasible(truck_id, customer_id, pos):
                cost = self._calculate_truck_insertion_cost(truck_id, customer_id, pos)
                options.append({
                    'type': 'truck',
                    'position': pos,
                    'cost': cost,
                    'prev_customer': truck_route[pos - 1] if pos > 0 else 0,
                    'next_customer': truck_route[pos] if pos < len(truck_route) else 0
                })
        return options

    def _get_drone_insertion_options(self, truck_id: int, customer_id: int) -> List[Dict]:
        """获取指定车辆对无人机路径的插入选项"""
        options = []
        customer = self.dyn_opt.customers[customer_id - 1]
        # 检查客户是否适合无人机服务
        if (customer.drone_eligible != 1 or
                abs(customer.demand) > self.dyn_opt.drone_max_capacity):
            return options
        # 遍历指定车辆对的无人机路径
        for trip_idx, trip in enumerate(self.dyn_opt.DRONE_Routes[truck_id].route):
            path = trip['path']
            # 尝试每个可能的插入位置
            for pos in range(1, len(path)):
                if self._is_drone_insertion_feasible(truck_id, trip_idx, customer_id, pos):
                    cost = self._calculate_drone_insertion_cost(truck_id, trip_idx, customer_id, pos)
                    options.append({
                        'type': 'drone',
                        'trip_index': trip_idx,
                        'position': pos,
                        'cost': cost,
                        'prev_customer': path[pos - 1],
                        'next_customer': path[pos]
                    })
        return options

    def _execute_insertion(self, truck_id: int, insertion_info: Dict) -> bool:
        """执行约束版插入操作"""
        try:
            customer_id = insertion_info['customer_id']
            option = insertion_info['option']
            if option['type'] == 'truck':
                return self._insert_customer_to_truck_at_position(
                    truck_id, customer_id, option['position'])
            elif option['type'] == 'drone':
                return self._insert_customer_to_drone_at_position(
                    truck_id, customer_id, option['trip_index'], option['position'])
            return False
        except Exception as e:
            print(f"       插入执行失败: {e}")
            return False

    def _is_truck_insertion_feasible(self, truck_id: int, customer_id: int, position: int) -> bool:
        """检查在指定车辆对的卡车中插入的可行性"""
        customer = self.dyn_opt.customers[customer_id - 1]
        truck = self.dyn_opt.TRUCK_Routes[truck_id]
        # 1. 载重约束检查
        if customer.demand > 0:
            if truck.current_load + customer.demand > truck.max_capacity:
                return False
        # 2. 时间窗约束检查（简化）
        return True

    def _is_drone_insertion_feasible(self, truck_id: int, trip_idx: int, customer_id: int, position: int) -> bool:
        """检查在指定车辆对的无人机中插入的可行性"""
        customer = self.dyn_opt.customers[customer_id - 1]
        trip = self.dyn_opt.DRONE_Routes[truck_id].route[trip_idx]
        # 1. 载重约束检查
        current_load = trip.get('current_load', 0)
        if current_load + abs(customer.demand) > self.dyn_opt.drone_max_capacity:
            return False
        # 2. 能耗约束检查（简化）
        return True

    def _insert_customer_to_truck_at_position(self, truck_id: int, customer_id: int, position: int) -> bool:
        """在指定车辆对的卡车路径指定位置插入客户"""
        try:
            self.dyn_opt.TRUCK_Routes[truck_id].Troute.insert(position, customer_id)
            # 更新客户信息
            self.dyn_opt.customers[customer_id - 1].service_by = ["tk", truck_id]
            # 更新时间矩阵
            self.dyn_opt.Update_visit_T(truck_id, position)
            return True
        except Exception as e:
            print(f"      ❌ 车辆对{truck_id}卡车插入失败: {e}")
            return False

    def _insert_customer_to_drone_at_position(self, truck_id: int, customer_id: int, trip_idx: int, position: int) -> bool:
        """在指定车辆对的无人机路径指定位置插入客户"""
        try:
            trip = self.dyn_opt.DRONE_Routes[truck_id].route[trip_idx]
            trip['path'].insert(position, customer_id)

            # 更新载重
            customer = self.dyn_opt.customers[customer_id - 1]
            if customer.demand > 0:
                trip['current_load'] = trip.get('current_load', 0) + customer.demand
                trip['current_load_delivery'] = trip.get('current_load_delivery', 0) + customer.demand
            else:
                trip['current_load_pickup'] = trip.get('current_load_pickup', 0) + abs(customer.demand)
            # 重新计算能耗
            first_node = trip['path'][0] - 1
            if hasattr(self.dyn_opt, 'Vist_T') and first_node >= 0:
                trip['energy'] = self.dyn_opt.calculate_energy(
                    self.dyn_opt.Vist_T[first_node][4], trip['path'], trip.get('current_load', 0))
            # 更新客户信息
            self.dyn_opt.customers[customer_id - 1].service_by = ["de", truck_id]
            return True
        except Exception as e:
            print(f"      ❌ 车辆对{truck_id}无人机插入失败: {e}")
            return False

    def _calculate_truck_insertion_cost(self, truck_id: int, customer_id: int, position: int) -> float:
        """计算在指定车辆对卡车中插入的成本"""
        truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
        if position == 0 or position >= len(truck_route):
            return float('inf')
        prev_node = truck_route[position - 1] if position > 0 else 0
        next_node = truck_route[position] if position < len(truck_route) else 0
        # 使用信息素指导评分（如果可用）
        if hasattr(self.dyn_opt, 'get_pheromone_guided_insertion_score'):
            # 计算基础距离成本
            if prev_node == 0:
                old_distance = self.dyn_opt.ALLdistanceTmatrix[0][next_node]
                new_distance = (self.dyn_opt.ALLdistanceTmatrix[0][customer_id] +
                                self.dyn_opt.ALLdistanceTmatrix[customer_id][next_node])
            elif next_node == 0:
                old_distance = self.dyn_opt.ALLdistanceTmatrix[prev_node][0]
                new_distance = (self.dyn_opt.ALLdistanceTmatrix[prev_node][customer_id] +
                                self.dyn_opt.ALLdistanceTmatrix[customer_id][0])
            else:
                old_distance = self.dyn_opt.ALLdistanceTmatrix[prev_node][next_node]
                new_distance = (self.dyn_opt.ALLdistanceTmatrix[prev_node][customer_id] +
                                self.dyn_opt.ALLdistanceTmatrix[customer_id][next_node])

            base_cost = (new_distance - old_distance) * self.dyn_opt.cost_truck
            score = self.dyn_opt.get_pheromone_guided_insertion_score(customer_id, prev_node, next_node, base_cost)
            return -score  # 负值，因为我们要最小化成本
        else:
            # 传统距离成本计算
            return self._calculate_basic_insertion_cost(prev_node, customer_id, next_node, 'truck')

    def _calculate_drone_insertion_cost(self, truck_id: int, trip_idx: int, customer_id: int, position: int) -> float:
        """计算在指定车辆对无人机中插入的成本"""
        trip = self.dyn_opt.DRONE_Routes[truck_id].route[trip_idx]
        path = trip['path']
        if position >= len(path):
            return float('inf')
        prev_node = path[position - 1]
        next_node = path[position]
        # 使用信息素指导评分
        if hasattr(self.dyn_opt, 'get_pheromone_guided_insertion_score'):
            old_distance = self.dyn_opt.ALLdistanceDmatrix[prev_node][next_node]
            new_distance = (self.dyn_opt.ALLdistanceDmatrix[prev_node][customer_id] +
                            self.dyn_opt.ALLdistanceDmatrix[customer_id][next_node])
            base_cost = (new_distance - old_distance) * self.dyn_opt.cost_drone
            score = self.dyn_opt.get_pheromone_guided_insertion_score(customer_id, prev_node, next_node, base_cost)
            return -score
        else:
            return self._calculate_basic_insertion_cost(prev_node, customer_id, next_node, 'drone')

    def _calculate_basic_insertion_cost(self, prev_node: int, customer_id: int, next_node: int, vehicle_type: str) -> float:
        """计算基础插入成本"""
        if vehicle_type == 'truck':
            distance_matrix = self.dyn_opt.ALLdistanceTmatrix
            unit_cost = self.dyn_opt.cost_truck
        else:
            distance_matrix = self.dyn_opt.ALLdistanceDmatrix
            unit_cost = self.dyn_opt.cost_drone
        old_distance = distance_matrix[prev_node][next_node]
        new_distance = distance_matrix[prev_node][customer_id] + distance_matrix[customer_id][next_node]
        return (new_distance - old_distance) * unit_cost

    def _create_new_drone_route(self, truck_id: int, candidate_customers: List[int]) -> List[int]:
        """
        为指定车辆对的候选客户创建新的无人机路径
        """
        if not candidate_customers:
            return []
        #只在指定车辆对内寻找起飞-回收节点对
        launch_retrieval_pairs = self._find_suitable_launch_retrieval_pairs(truck_id)
        for launch_node, retrieval_node in launch_retrieval_pairs:
            # 尝试构建包含尽可能多客户的路径
            route_customers = self._build_drone_route_with_customers(
                truck_id, launch_node, retrieval_node, candidate_customers)

            if route_customers:
                # 为指定车辆对创建新的无人机行程
                self._create_drone_trip(truck_id, launch_node, retrieval_node, route_customers)
                return route_customers
        return []

    def _find_suitable_launch_retrieval_pairs(self, truck_id: int) -> List[Tuple[int, int]]:
        """在指定车辆对内找到合适的起飞-回收节点对"""
        pairs = []
        truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
        # 获取当前车辆对已有的起飞和回收节点
        existing_launch = set()
        existing_retrieval = set()
        for trip in self.dyn_opt.DRONE_Routes[truck_id].route:
            existing_launch.add(trip['launch_node'])
            existing_retrieval.add(trip['retrieval_node'])
        # 只在当前车辆对的卡车路径中寻找节点对
        for i in range(1, len(truck_route) - 2):
            for j in range(i + 1, len(truck_route) - 1):
                launch_node = truck_route[i]
                retrieval_node = truck_route[j]
                # 验证节点属于当前车辆对
                if (self.dyn_opt.validate_customer_assignment(truck_id, launch_node) and
                        self.dyn_opt.validate_customer_assignment(truck_id, retrieval_node) and
                        launch_node not in existing_launch and
                        retrieval_node not in existing_retrieval):
                    pairs.append((launch_node, retrieval_node))
        return pairs

    def _build_drone_route_with_customers(self, truck_id: int, launch_node: int, retrieval_node: int, candidates: List[int]) -> List[int]:
        """在指定车辆对内构建包含候选客户的无人机路径"""
        route_customers = []
        current_load = 0
        current_energy = 0
        # 验证所有候选客户都属于当前车辆对
        vehicle_customers = self.dyn_opt.get_vehicle_customers(truck_id)
        valid_candidates = [c for c in candidates if c in vehicle_customers]

        if not valid_candidates:
            print(f"      ️ 没有属于车辆对{truck_id}的有效候选客户")
            return []
        # 按与起飞点的距离排序候选客户
        launch_customer = self.dyn_opt.customers[launch_node - 1]
        candidates_with_distance = []
        for customer_id in valid_candidates:
            customer = self.dyn_opt.customers[customer_id - 1]
            distance = math.sqrt((customer.xcoord - launch_customer.xcoord) ** 2 +
                                 (customer.ycoord - launch_customer.ycoord) ** 2)
            candidates_with_distance.append((customer_id, distance))
        # 按距离排序
        candidates_with_distance.sort(key=lambda x: x[1])
        # 逐个尝试添加客户
        for customer_id, _ in candidates_with_distance:
            customer = self.dyn_opt.customers[customer_id - 1]
            # 检查载重约束
            if current_load + abs(customer.demand) > self.dyn_opt.drone_max_capacity:
                continue
            # 构建临时路径并检查能耗约束
            temp_route = [launch_node] + route_customers + [customer_id, retrieval_node]
            temp_energy = self._calculate_drone_route_energy(temp_route, current_load + abs(customer.demand))
            if temp_energy <= self.dyn_opt.drone_max_battery:
                route_customers.append(customer_id)
                current_load += abs(customer.demand)
                current_energy = temp_energy
            else:
                break
        return route_customers

    def _calculate_drone_route_energy(self, route: List[int], load: float) -> float:
        """计算无人机路径的能耗"""
        total_energy = 0
        current_load = load
        for i in range(len(route) - 1):
            from_node = route[i] - 1 if route[i] > 0 else -1  # -1表示仓库
            to_node = route[i + 1] - 1 if route[i + 1] > 0 else -1
            # 计算飞行距离和时间
            if from_node >= 0 and to_node >= 0:
                from_customer = self.dyn_opt.customers[from_node]
                to_customer = self.dyn_opt.customers[to_node]
                distance = math.sqrt((from_customer.xcoord - to_customer.xcoord) ** 2 +
                                     (from_customer.ycoord - to_customer.ycoord) ** 2)
            else:
                # 涉及仓库的距离计算
                distance = 0  # 简化处理
            flight_time = distance / self.dyn_opt.drone_speed
            flight_energy = (current_load + self.dyn_opt.drone_weight) * flight_time * self.dyn_opt.energy_fight
            total_energy += flight_energy
            # 如果不是最后一个节点，加上服务能耗
            if i < len(route) - 2:
                service_energy = (current_load + self.dyn_opt.drone_weight) * self.dyn_opt.service_time * self.dyn_opt.energy_service
                total_energy += service_energy
                # 更新载重
                if to_node >= 0:
                    customer = self.dyn_opt.customers[to_node]
                    if customer.demand > 0:
                        current_load -= customer.demand
                    else:
                        current_load += abs(customer.demand)
        return total_energy

    def _create_drone_trip(self, truck_id: int, launch_node: int, retrieval_node: int, customers: List[int]):
        """为指定车辆对创建新的无人机行程"""
        path = [launch_node] + customers + [retrieval_node]
        # 计算载重（仅考虑当前车辆对的客户）
        total_load = sum(abs(self.dyn_opt.customers[c - 1].demand) for c in customers
                         if (c <= len(self.dyn_opt.customers) and
                             self.dyn_opt.customers[c - 1].demand > 0 and
                             self.dyn_opt.validate_customer_assignment(truck_id, c)))
        # 计算能耗
        energy = self._calculate_drone_route_energy(path, total_load)
        # 添加到指定车辆对的无人机路径
        self.dyn_opt.DRONE_Routes[truck_id].add_trip(launch_node, retrieval_node, path, energy)
        # 更新客户服务信息（确保指向正确的车辆对）
        for customer_id in customers:
            if self.dyn_opt.validate_customer_assignment(truck_id, customer_id):
                self.dyn_opt.customers[customer_id - 1].service_by = ["de", truck_id]

    def emergency_repair(self, truck_id: int, delete_list: List[int]) -> bool:
        """
        应急修复策略：只在指定车辆对内进行应急修复
        """
        print(f"执行车辆对{truck_id}的应急修复...")
        if not delete_list:
            return True
        # 验证所有客户都属于指定车辆对
        vehicle_customers = self.dyn_opt.get_vehicle_customers(truck_id)
        valid_customers = [c for c in delete_list if c in vehicle_customers]
        if len(valid_customers) != len(delete_list):
            invalid_customers = [c for c in delete_list if c not in vehicle_customers]
            print(f"发现{len(invalid_customers)}个不属于车辆对{truck_id}的客户: {invalid_customers}")
        inserted_count = 0
        for customer_id in valid_customers:
            try:
                # 关键：只尝试插入到指定的truck_id路径中
                truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
                insert_pos = len(truck_route) - 1  # 在返回仓库前插入
                # 验证插入位置的合理性
                if insert_pos > 0:
                    truck_route.insert(insert_pos, customer_id)
                    # 更新客户服务信息（确保指向正确的车辆对）
                    self.dyn_opt.customers[customer_id - 1].service_by = ["tk", truck_id]
                    # 更新时间矩阵
                    try:
                        self.dyn_opt.Update_visit_T(truck_id, insert_pos)
                    except Exception as time_error:
                        print(f"更新时间矩阵出错: {time_error}")
                    inserted_count += 1
                    print(f"          应急插入客户{customer_id}到车辆对{truck_id}卡车路径")
                else:
                    print(f"          车辆对{truck_id}路径插入位置无效")
            except Exception as e:
                print(f"          应急插入客户{customer_id}到车辆对{truck_id}失败: {e}")
        success_rate = (inserted_count / len(valid_customers)) * 100 if valid_customers else 100
        print(f"       车辆对{truck_id}应急修复完成: {inserted_count}/{len(valid_customers)} ({success_rate:.1f}%)")
        return inserted_count > 0

    def _enforce_package_constraints(self, vehicle_id: int, customers_to_repair: list) -> list:
        """
        在修复过程中强制执行包裹约束
        """
        vehicle_customers = self.dyn_opt.get_vehicle_customers(vehicle_id)
        # 过滤：只允许修复属于当前车辆对的客户
        valid_customers = []
        invalid_customers = []
        for customer_id in customers_to_repair:
            if customer_id in vehicle_customers:
                valid_customers.append(customer_id)
            else:
                invalid_customers.append(customer_id)
                actual_vehicle = self.dyn_opt.get_customer_vehicle(customer_id)
                print(f"        拒绝修复客户{customer_id}：属于车辆对{actual_vehicle}，不属于当前车辆对{vehicle_id}")
        if invalid_customers:
            print(f"        过滤了{len(invalid_customers)}个跨车辆对的客户")
        return valid_customers

# ==================== 完整可行性修复算子实现 ====================
class FeasibilityRepairOperators:
    """
    改进版可行性修复算子集合类 - 解决无限循环和修复效果问题
    直接替换原有实现，保持接口兼容性
    """
    def __init__(self, dynamic_opt_instance):
        self.dyn_opt = dynamic_opt_instance
        self.repair_stats = {
            'dlro_calls': 0, 'dlro_success': 0,
            'dero_calls': 0, 'dero_success': 0,
            'tlro_calls': 0, 'tlro_success': 0,
            'twro_calls': 0, 'twro_success': 0
        }

        #  改进的修复参数
        self.max_repair_attempts = 3  # 大幅减少最大尝试次数
        self.violation_tolerance = 0.1  # 违反容忍度
        self.enable_aggressive_repair = True  # 启用激进修复模式
        self.debug_mode = False  # 调试模式开关

    def check_and_repair_feasibility(self, truck_id: int) -> bool:
        """
         改进版可行性检查和修复 - 彻底解决无限循环问题
        """
        if self.debug_mode:
            print(f"🔧 开始车辆对{truck_id}可行性检查（改进版）...")

        repair_attempts = 0
        max_attempts = self.max_repair_attempts
        overall_success = True

        #  关键改进：记录已处理的违反类型，防止无限循环
        processed_violations = set()
        consecutive_same_violations = 0
        last_violation_signature = None

        while repair_attempts < max_attempts:
            violations_found = False

            #  一次性检查所有违反类型
            violation_summary = self._comprehensive_violation_check(truck_id)

            if not violation_summary:
                if self.debug_mode:
                    print(f"    车辆对{truck_id}所有约束都已满足")
                break

            #  生成违反签名，检测是否陷入循环
            current_signature = self._generate_violation_signature(violation_summary)
            if current_signature == last_violation_signature:
                consecutive_same_violations += 1
                if consecutive_same_violations >= 2:  # 连续2次相同违反就跳出
                    if self.debug_mode:
                        print(f"   ️ 检测到循环违反，启动激进修复...")
                    break
            else:
                consecutive_same_violations = 0

            last_violation_signature = current_signature

            #  按优先级处理违反（一次只处理一种类型）
            repair_priority = ['truck_load', 'drone_load', 'drone_energy', 'time_window']

            repair_performed = False
            for violation_type in repair_priority:
                if violation_type in violation_summary:
                    violation_key = f"{violation_type}_{truck_id}_{repair_attempts}"

                    if self.debug_mode:
                        print(f"   处理{violation_type}违反（轮次{repair_attempts + 1}）...")

                    # 执行修复
                    success = self._repair_violation_by_type(
                        truck_id, violation_type, violation_summary[violation_type])

                    processed_violations.add(violation_key)
                    violations_found = True
                    repair_performed = True

                    if success:
                        if self.debug_mode:
                            print(f"      {violation_type}修复成功，重新计算状态...")
                        #  关键：修复后立即重新计算所有状态
                        self._recalculate_all_states(truck_id)
                    else:
                        if self.debug_mode:
                            print(f"      {violation_type}修复失败")
                        overall_success = False

                    #  每次修复后立即跳出，重新检查
                    break

            if not repair_performed:
                if self.debug_mode:
                    print(f"   ️ 无新的违反需要处理")
                break

            repair_attempts += 1

        #  如果常规修复失败或检测到循环，启动激进修复
        if (repair_attempts >= max_attempts or consecutive_same_violations >= 2) and self.enable_aggressive_repair:
            if self.debug_mode:
                print(f"    启动激进修复模式...")
            aggressive_success = self._aggressive_repair_mode(truck_id)
            if aggressive_success:
                overall_success = True
                if self.debug_mode:
                    print(f"    激进修复成功")
            else:
                overall_success = False
                if self.debug_mode:
                    print(f"    激进修复失败")

        return overall_success

    def _generate_violation_signature(self, violation_summary: Dict) -> str:
        """ 生成违反签名，用于检测循环"""
        signature_parts = []
        for violation_type, violations in violation_summary.items():
            if violations:
                signature_parts.append(f"{violation_type}:{len(violations)}")
        return "|".join(sorted(signature_parts))

    def _comprehensive_violation_check(self, truck_id: int) -> Dict:
        """ 全面的违反检查 - 一次性检查所有类型"""
        violations = {}

        try:
            # 1. 检查卡车载重违反
            truck_violations = self._check_truck_load_violations_detailed(truck_id)
            if truck_violations:
                violations['truck_load'] = truck_violations

            # 2. 检查无人机载重违反
            drone_load_violations = self._check_drone_load_violations_detailed(truck_id)
            if drone_load_violations:
                violations['drone_load'] = drone_load_violations

            # 3. 检查无人机能耗违反
            energy_violations = self._check_drone_energy_violations_detailed(truck_id)
            if energy_violations:
                violations['drone_energy'] = energy_violations

            # 4. 检查时间窗口违反
            time_violations = self._check_time_window_violations_detailed(truck_id)
            if time_violations:
                violations['time_window'] = time_violations

        except Exception as e:
            if self.debug_mode:
                print(f"      违反检查出错: {e}")

        return violations

    def _check_truck_load_violations_detailed(self, truck_id: int) -> List[Dict]:
        """ 详细的卡车载重检查"""
        violations = []
        try:
            truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
            current_load = self.dyn_opt.TRUCK_Routes[truck_id].current_load

            for i, customer_id in enumerate(truck_route[1:-1], 1):
                if customer_id <= len(self.dyn_opt.customers):
                    customer = self.dyn_opt.customers[customer_id - 1]

                    # 取货操作检查
                    if customer.demand < 0:
                        projected_load = current_load + abs(customer.demand)
                        if projected_load > self.dyn_opt.truck_max_capacity:
                            violations.append({
                                'customer_id': customer_id,
                                'position': i,
                                'current_load': current_load,
                                'projected_load': projected_load,
                                'excess': projected_load - self.dyn_opt.truck_max_capacity
                            })

                    current_load -= customer.demand

        except Exception as e:
            if self.debug_mode:
                print(f"     卡车载重检查出错: {e}")

        return violations

    def _check_drone_load_violations_detailed(self, truck_id: int) -> List[Dict]:
        """ 详细的无人机载重检查"""
        violations = []
        try:
            for trip_idx, trip in enumerate(self.dyn_opt.DRONE_Routes[truck_id].route):
                # 检查总载重
                total_load = trip.get('current_load', 0)
                if total_load > self.dyn_opt.drone_max_capacity:
                    violations.append({
                        'trip_index': trip_idx,
                        'violation_type': 'launch_overload',
                        'current_load': total_load,
                        'excess': total_load - self.dyn_opt.drone_max_capacity
                    })

                # 检查飞行过程载重变化
                flight_violations = self._check_flight_load_progression(trip, trip_idx)
                violations.extend(flight_violations)

        except Exception as e:
            if self.debug_mode:
                print(f"     无人机载重检查出错: {e}")

        return violations

    def _check_flight_load_progression(self, trip: Dict, trip_idx: int) -> List[Dict]:
        """ 检查飞行过程中的载重变化"""
        violations = []
        try:
            path = trip['path']
            current_load = trip.get('initial_load', 0)

            for i, customer_id in enumerate(path[1:-1], 1):
                if customer_id <= len(self.dyn_opt.customers):
                    customer = self.dyn_opt.customers[customer_id - 1]

                    if customer.demand > 0:
                        current_load -= customer.demand
                    else:
                        current_load += abs(customer.demand)
                        if current_load > self.dyn_opt.drone_max_capacity:
                            violations.append({
                                'trip_index': trip_idx,
                                'violation_type': 'flight_overload',
                                'customer_id': customer_id,
                                'position': i,
                                'load_at_violation': current_load,
                                'excess': current_load - self.dyn_opt.drone_max_capacity
                            })

        except Exception as e:
            if self.debug_mode:
                print(f"     飞行载重检查出错: {e}")

        return violations

    def _check_drone_energy_violations_detailed(self, truck_id: int) -> List[Dict]:
        """ 详细的能耗检查"""
        violations = []
        try:
            for trip_idx, trip in enumerate(self.dyn_opt.DRONE_Routes[truck_id].route):
                total_energy = trip.get('energy', 0)
                if total_energy > self.dyn_opt.drone_max_battery:
                    violations.append({
                        'trip_index': trip_idx,
                        'total_energy': total_energy,
                        'excess_energy': total_energy - self.dyn_opt.drone_max_battery,
                        'path': trip['path']
                    })

        except Exception as e:
            if self.debug_mode:
                print(f"     无人机能耗检查出错: {e}")

        return violations

    def _check_time_window_violations_detailed(self, truck_id: int) -> List[Dict]:
        """ 详细的时间窗口检查 - 只检查显著违反"""
        violations = []
        try:
            # 检查卡车路径时间窗口
            for customer_id in self.dyn_opt.TRUCK_Routes[truck_id].Troute[1:-1]:
                if customer_id <= len(self.dyn_opt.customers):
                    customer = self.dyn_opt.customers[customer_id - 1]
                    arrival_time = getattr(customer, 'arrive_truck', customer.start_time)

                    # 只记录显著的早到（超过容忍度）
                    if arrival_time < customer.start_time:
                        deviation = customer.start_time - arrival_time
                        if deviation > self.violation_tolerance:
                            violations.append({
                                'customer_id': customer_id,
                                'vehicle_type': 'truck',
                                'violation_type': 'early',
                                'deviation': deviation
                            })
                    # 晚到始终记录
                    elif arrival_time > customer.end_time:
                        deviation = arrival_time - customer.end_time
                        violations.append({
                            'customer_id': customer_id,
                            'vehicle_type': 'truck',
                            'violation_type': 'late',
                            'deviation': deviation
                        })

            # 检查无人机路径时间窗口
            for trip_idx, trip in enumerate(self.dyn_opt.DRONE_Routes[truck_id].route):
                for customer_id in trip['path'][1:-1]:
                    if customer_id <= len(self.dyn_opt.customers):
                        customer = self.dyn_opt.customers[customer_id - 1]
                        arrival_time = getattr(customer, 'arrive_drone', customer.start_time)

                        if arrival_time < customer.start_time:
                            deviation = customer.start_time - arrival_time
                            if deviation > self.violation_tolerance:
                                violations.append({
                                    'customer_id': customer_id,
                                    'vehicle_type': 'drone',
                                    'trip_index': trip_idx,
                                    'violation_type': 'early',
                                    'deviation': deviation
                                })
                        elif arrival_time > customer.end_time:
                            deviation = arrival_time - customer.end_time
                            violations.append({
                                'customer_id': customer_id,
                                'vehicle_type': 'drone',
                                'trip_index': trip_idx,
                                'violation_type': 'late',
                                'deviation': deviation
                            })

        except Exception as e:
            if self.debug_mode:
                print(f"     时间窗口检查出错: {e}")

        return violations

    def _repair_violation_by_type(self, truck_id: int, violation_type: str, violations: List[Dict]) -> bool:
        """ 根据违反类型执行对应的修复"""
        try:
            if violation_type == 'truck_load':
                return self._repair_truck_load_violations(truck_id, violations)
            elif violation_type == 'drone_load':
                return self._repair_drone_load_violations(truck_id, violations)
            elif violation_type == 'drone_energy':
                return self._repair_drone_energy_violations(truck_id, violations)
            elif violation_type == 'time_window':
                return self._repair_time_window_violations(truck_id, violations)
            else:
                if self.debug_mode:
                    print(f"     ❌ 未知的违反类型: {violation_type}")
                return False
        except Exception as e:
            if self.debug_mode:
                print(f"     ❌ 修复{violation_type}时出错: {e}")
            return False

    def _repair_truck_load_violations(self, truck_id: int, violations: List[Dict]) -> bool:
        """ 修复卡车载重违反"""
        self.repair_stats['tlro_calls'] += 1
        success = True

        try:
            # 按严重程度排序
            violations.sort(key=lambda x: x.get('excess', 0), reverse=True)

            for violation in violations:
                customer_id = violation['customer_id']
                position = violation['position']

                # 尝试将客户移动到路径末尾
                truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
                if customer_id in truck_route:
                    truck_route.remove(customer_id)
                    # 插入到返回仓库前
                    insert_pos = len(truck_route) - 1
                    truck_route.insert(insert_pos, customer_id)
                    if self.debug_mode:
                        print(f"       客户{customer_id}移至路径末尾")
                else:
                    success = False

        except Exception as e:
            if self.debug_mode:
                print(f"       卡车载重修复出错: {e}")
            success = False

        if success:
            self.repair_stats['tlro_success'] += 1

        return success

    def _repair_drone_load_violations(self, truck_id: int, violations: List[Dict]) -> bool:
        """ 修复无人机载重违反"""
        self.repair_stats['dlro_calls'] += 1
        success = True

        try:
            for violation in violations:
                trip_idx = violation['trip_index']

                if violation['violation_type'] == 'launch_overload':
                    success &= self._fix_launch_overload(truck_id, trip_idx)
                elif violation['violation_type'] == 'flight_overload':
                    success &= self._fix_flight_overload(truck_id, trip_idx, violation)

        except Exception as e:
            if self.debug_mode:
                print(f"       无人机载重修复出错: {e}")
            success = False

        if success:
            self.repair_stats['dlro_success'] += 1

        return success

    def _fix_launch_overload(self, truck_id: int, trip_idx: int) -> bool:
        """ 修复起飞过载"""
        try:
            if trip_idx >= len(self.dyn_opt.DRONE_Routes[truck_id].route):
                return False

            trip = self.dyn_opt.DRONE_Routes[truck_id].route[trip_idx]
            path = trip['path']

            if len(path) <= 2:  # 只有起终点
                return True

            # 简化策略：移除一半客户到卡车
            customers_to_remove = path[1:len(path) // 2]

            for customer_id in customers_to_remove:
                if customer_id in path:
                    path.remove(customer_id)
                    # 添加到卡车路径末尾
                    truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
                    insert_pos = len(truck_route) - 1
                    truck_route.insert(insert_pos, customer_id)

                    # 更新客户服务信息
                    if customer_id <= len(self.dyn_opt.customers):
                        self.dyn_opt.customers[customer_id - 1].service_by = ["tk", truck_id]

            # 重新计算载重
            self._recalculate_trip_load(trip)
            if self.debug_mode:
                print(f"       移除{len(customers_to_remove)}个客户到卡车")
            return True

        except Exception as e:
            if self.debug_mode:
                print(f"       起飞过载修复出错: {e}")
            return False

    def _fix_flight_overload(self, truck_id: int, trip_idx: int, violation: Dict) -> bool:
        """ 修复飞行中过载"""
        try:
            customer_id = violation.get('customer_id')
            if not customer_id:
                return False

            trip = self.dyn_opt.DRONE_Routes[truck_id].route[trip_idx]

            # 移除导致过载的客户
            if customer_id in trip['path']:
                trip['path'].remove(customer_id)

                # 添加到卡车路径
                truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
                insert_pos = len(truck_route) - 1
                truck_route.insert(insert_pos, customer_id)

                # 更新客户服务信息
                if customer_id <= len(self.dyn_opt.customers):
                    self.dyn_opt.customers[customer_id - 1].service_by = ["tk", truck_id]

                # 重新计算载重
                self._recalculate_trip_load(trip)
                if self.debug_mode:
                    print(f"       移除过载客户{customer_id}到卡车")
                return True

        except Exception as e:
            if self.debug_mode:
                print(f"       飞行过载修复出错: {e}")
            return False

        return False

    def _repair_drone_energy_violations(self, truck_id: int, violations: List[Dict]) -> bool:
        """ 修复无人机能耗违反"""
        self.repair_stats['dero_calls'] += 1
        success = True

        try:
            for violation in violations:
                trip_idx = violation['trip_index']
                path = violation['path']

                if len(path) <= 2:
                    continue

                # 简化策略：移除路径中一半的客户
                trip = self.dyn_opt.DRONE_Routes[truck_id].route[trip_idx]
                customers_to_remove = path[1:len(path) // 2 + 1]

                for customer_id in customers_to_remove:
                    if customer_id in trip['path']:
                        trip['path'].remove(customer_id)

                        # 添加到卡车路径
                        truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
                        insert_pos = len(truck_route) - 1
                        truck_route.insert(insert_pos, customer_id)

                        # 更新客户服务信息
                        if customer_id <= len(self.dyn_opt.customers):
                            self.dyn_opt.customers[customer_id - 1].service_by = ["tk", truck_id]

                # 重新计算能耗
                if len(trip['path']) > 2:
                    self._recalculate_trip_energy(trip)

                if self.debug_mode:
                    print(f"       因能耗约束移除{len(customers_to_remove)}个客户")

        except Exception as e:
            if self.debug_mode:
                print(f"       能耗修复出错: {e}")
            success = False

        if success:
            self.repair_stats['dero_success'] += 1

        return success

    def _repair_time_window_violations(self, truck_id: int, violations: List[Dict]) -> bool:
        """ 改进的时间窗口修复 - 减少不必要的等待时间设置"""
        self.repair_stats['twro_calls'] += 1
        success = True

        try:
            # 只处理严重的时间窗口违反
            serious_violations = [v for v in violations if v.get('deviation', 0) > self.violation_tolerance]

            if not serious_violations:
                self.repair_stats['twro_success'] += 1
                return True

            # 分类处理
            early_violations = [v for v in serious_violations if v['violation_type'] == 'early']
            late_violations = [v for v in serious_violations if v['violation_type'] == 'late']

            # 处理早到 - 设置等待时间但不重复设置
            processed_early = set()
            for violation in early_violations:
                customer_id = violation['customer_id']
                if customer_id in processed_early:
                    continue

                customer = self.dyn_opt.customers[customer_id - 1]
                deviation = violation['deviation']

                # 设置等待时间
                customer.service_begin = customer.start_time
                processed_early.add(customer_id)

                if self.debug_mode:
                    print(f"       {violation['vehicle_type']}在客户{customer_id}处等待{deviation:.2f}时间单位")

                # 如果是无人机，增加悬停能耗
                if violation['vehicle_type'] == 'drone':
                    self._add_hovering_energy(truck_id, violation.get('trip_index'), customer_id, deviation)

            # 处理晚到 - 简化策略：移动到路径更早位置
            for violation in late_violations[:3]:  # 限制处理数量
                customer_id = violation['customer_id']

                if violation['vehicle_type'] == 'truck':
                    self._move_customer_earlier_in_truck(truck_id, customer_id)
                else:
                    self._move_customer_earlier_in_drone(truck_id, violation.get('trip_index'), customer_id)

        except Exception as e:
            if self.debug_mode:
                print(f"       时间窗口修复出错: {e}")
            success = False

        if success:
            self.repair_stats['twro_success'] += 1

        return success

    def _move_customer_earlier_in_truck(self, truck_id: int, customer_id: int) -> bool:
        """ 将卡车客户移到更早位置"""
        try:
            truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
            if customer_id in truck_route:
                current_pos = truck_route.index(customer_id)
                if current_pos > 1:  # 可以向前移动
                    truck_route.remove(customer_id)
                    new_pos = max(1, current_pos - 2)  # 向前移动2个位置
                    truck_route.insert(new_pos, customer_id)
                    if self.debug_mode:
                        print(f"       卡车客户{customer_id}从位置{current_pos}移至{new_pos}")
                    return True
        except Exception as e:
            if self.debug_mode:
                print(f"       移动卡车客户出错: {e}")
        return False

    def _move_customer_earlier_in_drone(self, truck_id: int, trip_idx: int, customer_id: int) -> bool:
        """ 将无人机客户移到更早位置"""
        try:
            if trip_idx is None or trip_idx >= len(self.dyn_opt.DRONE_Routes[truck_id].route):
                return False

            trip = self.dyn_opt.DRONE_Routes[truck_id].route[trip_idx]
            path = trip['path']

            if customer_id in path:
                current_pos = path.index(customer_id)
                if current_pos > 1:
                    path.remove(customer_id)
                    new_pos = max(1, current_pos - 1)
                    path.insert(new_pos, customer_id)
                    if self.debug_mode:
                        print(f"       无人机客户{customer_id}在路径内向前移动")
                    return True
        except Exception as e:
            if self.debug_mode:
                print(f"       移动无人机客户出错: {e}")
        return False

    def _add_hovering_energy(self, truck_id: int, trip_idx: Optional[int], customer_id: int, wait_time: float):
        """ 添加无人机悬停能耗"""
        try:
            if trip_idx is not None and trip_idx < len(self.dyn_opt.DRONE_Routes[truck_id].route):
                trip = self.dyn_opt.DRONE_Routes[truck_id].route[trip_idx]
                additional_energy = wait_time * self.dyn_opt.energy_hover * self.dyn_opt.drone_weight
                trip['energy'] = trip.get('energy', 0) + additional_energy
        except Exception as e:
            if self.debug_mode:
                print(f"       添加悬停能耗出错: {e}")

    def _aggressive_repair_mode(self, truck_id: int) -> bool:
        """ 激进修复模式 - 最后的修复手段"""
        if self.debug_mode:
            print(f"    执行车辆对{truck_id}激进修复...")

        try:
            # 1. 移除所有有问题的无人机任务
            trips_to_remove = []
            for trip_idx, trip in enumerate(self.dyn_opt.DRONE_Routes[truck_id].route):
                if (trip.get('energy', 0) > self.dyn_opt.drone_max_battery or
                        trip.get('current_load', 0) > self.dyn_opt.drone_max_capacity):
                    trips_to_remove.append(trip_idx)

            # 逆序删除
            for trip_idx in reversed(trips_to_remove):
                removed_trip = self.dyn_opt.DRONE_Routes[truck_id].route.pop(trip_idx)
                customers_to_reassign = removed_trip['path'][1:-1]

                # 重新分配到卡车
                truck_route = self.dyn_opt.TRUCK_Routes[truck_id].Troute
                insert_pos = len(truck_route) - 1

                for customer_id in customers_to_reassign:
                    truck_route.insert(insert_pos, customer_id)
                    insert_pos += 1
                    # 更新服务信息
                    if customer_id <= len(self.dyn_opt.customers):
                        self.dyn_opt.customers[customer_id - 1].service_by = ["tk", truck_id]

                if self.debug_mode:
                    print(f"     移除无人机任务{trip_idx}，重新分配{len(customers_to_reassign)}个客户")

            # 2. 重新计算所有状态
            self._recalculate_all_states(truck_id)

            # 3. 设置所有早到客户的等待时间
            for customer in self.dyn_opt.customers:
                if (hasattr(customer, 'service_by') and
                        customer.service_by and
                        customer.service_by[1] == truck_id):

                    arrival_time = getattr(customer, 'arrive_truck', customer.start_time)
                    if arrival_time < customer.start_time:
                        customer.service_begin = customer.start_time

            if self.debug_mode:
                print(f"      激进修复完成")
            return True

        except Exception as e:
            if self.debug_mode:
                print(f"      激进修复失败: {e}")
            return False

    def _recalculate_all_states(self, truck_id: int):
        """ 重新计算车辆所有状态 - 关键的状态同步方法"""
        try:
            # 1. 重新计算卡车时间
            if len(self.dyn_opt.TRUCK_Routes[truck_id].Troute) > 2:
                self.dyn_opt.Update_visit_T(truck_id, 1)

            # 2. 重新计算所有无人机任务的载重和能耗
            for trip in self.dyn_opt.DRONE_Routes[truck_id].route:
                if len(trip['path']) > 2:
                    # 重新计算载重
                    self._recalculate_trip_load(trip)
                    # 重新计算能耗
                    self._recalculate_trip_energy(trip)

            # 3. 更新客户服务时间
            self._update_customer_service_times(truck_id)

        except Exception as e:
            if self.debug_mode:
                print(f"       状态重新计算出错: {e}")

    def _recalculate_trip_load(self, trip: Dict):
        """ 重新计算trip载重"""
        try:
            total_delivery = 0
            total_pickup = 0
            for customer_id in trip['path'][1:-1]:
                if customer_id <= len(self.dyn_opt.customers):
                    customer = self.dyn_opt.customers[customer_id - 1]
                    if customer.demand > 0:
                        total_delivery += customer.demand
                    else:
                        total_pickup += abs(customer.demand)
            trip['current_load'] = total_delivery
            trip['current_load_delivery'] = total_delivery
            trip['current_load_pickup'] = total_pickup
            trip['initial_load'] = total_delivery
        except Exception as e:
            if self.debug_mode:
                print(f"       载重重新计算出错: {e}")

    def _recalculate_trip_energy(self, trip: Dict):
        """ 重新计算trip能耗"""
        try:
            if len(trip['path']) <= 2:
                trip['energy'] = 0
                return

            first_node_idx = trip['path'][0] - 1
            if first_node_idx >= 0 and hasattr(self.dyn_opt, 'Vist_T'):
                try:
                    trip['energy'] = self.dyn_opt.calculate_energy(
                        self.dyn_opt.Vist_T[first_node_idx][4],
                        trip['path'],
                        trip.get('current_load', 0)
                    )
                except:
                    # 如果计算失败，设置一个保守的能耗值
                    trip['energy'] = self.dyn_opt.drone_max_battery * 0.9

        except Exception as e:
            if self.debug_mode:
                print(f"       能耗重新计算出错: {e}")

    def _update_customer_service_times(self, truck_id: int):
        """ 更新客户服务时间"""
        try:
            # 更新所有客户的service_begin时间
            vehicle_customers = self.dyn_opt.get_vehicle_customers(truck_id)

            for customer_id in vehicle_customers:
                if customer_id <= len(self.dyn_opt.customers):
                    customer = self.dyn_opt.customers[customer_id - 1]

                    if hasattr(customer, 'service_by') and customer.service_by:
                        if customer.service_by[0] == "tk":
                            customer.service_begin = getattr(customer, 'arrive_truck', customer.start_time)
                        elif customer.service_by[0] == "de":
                            customer.service_begin = getattr(customer, 'arrive_drone', customer.start_time)

        except Exception as e:
            if self.debug_mode:
                print(f"       服务时间更新出错: {e}")

    # ==================== 原有接口方法（保持兼容性） ====================

    def drone_load_repair_operator(self, truck_id: int) -> bool:
        """保持原有接口兼容性"""
        violations = self._check_drone_load_violations_detailed(truck_id)
        if violations:
            return self._repair_drone_load_violations(truck_id, violations)
        return True

    def drone_energy_repair_operator(self, truck_id: int) -> bool:
        """保持原有接口兼容性"""
        violations = self._check_drone_energy_violations_detailed(truck_id)
        if violations:
            return self._repair_drone_energy_violations(truck_id, violations)
        return True

    def truck_load_repair_operator(self, truck_id: int) -> bool:
        """保持原有接口兼容性"""
        violations = self._check_truck_load_violations_detailed(truck_id)
        if violations:
            return self._repair_truck_load_violations(truck_id, violations)
        return True

    def time_window_repair_operator(self, truck_id: int) -> bool:
        """保持原有接口兼容性"""
        violations = self._check_time_window_violations_detailed(truck_id)
        if violations:
            return self._repair_time_window_violations(truck_id, violations)
        return True

    def print_repair_statistics(self):
        """打印改进版修复统计信息"""
        print("\n" + "=" * 50)
        print("改进版可行性修复算子统计:")
        print("=" * 50)

        total_calls = sum(self.repair_stats[f'{op}_calls'] for op in ['dlro', 'dero', 'tlro', 'twro'])
        total_success = sum(self.repair_stats[f'{op}_success'] for op in ['dlro', 'dero', 'tlro', 'twro'])

        for op_type in ['dlro', 'dero', 'tlro', 'twro']:
            calls = self.repair_stats[f'{op_type}_calls']
            success = self.repair_stats[f'{op_type}_success']
            success_rate = (success / calls * 100) if calls > 0 else 0
            op_name = {
                'dlro': 'DLRO (无人机载重)',
                'dero': 'DERO (无人机能耗)',
                'tlro': 'TLRO (卡车载重)',
                'twro': 'TWRO (时间窗口)'
            }[op_type]
            print(f"{op_name}: {calls}次调用, {success}次成功 ({success_rate:.1f}%)")

        overall_success_rate = (total_success / total_calls * 100) if total_calls > 0 else 0
        print("-" * 50)
        print(f"总体成功率: {total_success}/{total_calls} ({overall_success_rate:.1f}%)")
        print("=" * 50)

class Dynamic_Optimization:
    def __init__(self, truck_routes, drone_routes, clusters, customers, min_delete, max_delete, truck_max_capacity,
                 truck_speed, drone_speed, drone_weight, drone_max_capacity, drone_max_battery, service_time,
                 energy_fight, energy_service, energy_hover, cost_truck, cost_drone, ALLdistanceTmatrix,
                 ALLdistanceDmatrix, Initial_solution, Current_solution, Best_solution, Copy_solution):
        self.TRUCK_Routes = truck_routes    #卡车路径
        self.DRONE_Routes = drone_routes    #无人机路径
        self.clusters = clusters            #聚类信息
        self.customers = customers          #客户列表
        self.cnum=len(self.customers)       #客户数量
        self.min_delete=min_delete          #最小删除客户比例
        self.max_delete=max_delete          #最大删除客户比例
        self.truck_max_capacity=truck_max_capacity   #卡车载量
        self.truck_speed=truck_speed                 #卡车速度
        self.drone_speed=drone_speed                 #无人机速度
        self.drone_weight=drone_weight               #无人机重量
        self.drone_max_capacity=drone_max_capacity   #无人机最大载重
        self.drone_max_battery = drone_max_battery   #无人机最大续航
        self.service_time=service_time               #服务时间
        self.energy_fight=energy_fight
        self.energy_service=energy_service
        self.energy_hover=energy_hover
        self.cost_truck=cost_truck
        self.cost_drone=cost_drone
        self.ALLdistanceTmatrix=ALLdistanceTmatrix
        self.ALLdistanceDmatrix=ALLdistanceDmatrix

        self.Initial_solution=Initial_solution
        self.Current_solution=Current_solution
        self.Best_solution=Best_solution
        self.Copy_solution=Copy_solution

        #初始化修补算子
        self.repair_ops = RepairOperators(self)
        # 初始化摧毁算子
        self.destroy_ops = DestroyOperators(self)
        # 可行性修复算子
        try:
            self.feasibility_repair_ops = FeasibilityRepairOperators(self)
            self.enable_feasibility_check = True
            print(" 可行性修复算子初始化成功")
        except ImportError:
            print(" 可行性修复算子未找到，将跳过可行性检查")
            self.feasibility_repair_ops = None
            self.enable_feasibility_check = False

        # 阶段统计跟踪
        self.stage_statistics = {
            'total_stages': 0,
            'successful_services': 0,
            'failed_services': 0,
            'reconstructions': 0,
            'cost_improvements': 0
        }

        # ==================== 信息素机制初始化 ====================
        self.pheromone_matrix = None  # 信息素矩阵
        self.pheromone_alpha = 0.6  # 信息素影响因子 α
        self.pheromone_beta = 0.4  # 距离影响因子 β
        self.pheromone_learning_rate = 0.15  # 信息素学习率 ε
        self.pheromone_evaporation_rate = 0.02  # 信息素挥发率 ρ
        self.pheromone_min = 0.01  # 最小信息素值
        self.pheromone_max = 10.0  # 最大信息素值
        self.pheromone_initial = 1.0  # 初始信息素值
        self._pheromone_update_counter = 0  # 信息素更新计数器

        # 初始化信息素矩阵
        self.initialize_pheromone_matrix()
        # 在这里添加车辆对约束管理的初始化
        print(" 初始化车辆对约束管理...")
        self.vehicle_customer_assignment = {}  # {truck_id: set(customer_ids)}
        self.customer_to_vehicle = {}  # {customer_id: truck_id}
        self.vehicle_initial_packages = {}  # {truck_id: {'delivery': set(), 'pickup': set()}}
        print("    约束管理属性初始化完成")

        # ==================== 局部搜索 ====================
        self.theta = 0.08  # 质量阈值，决定何时启动局部搜索
        self.local_search_max_no_improve = 100  # 局部搜索停止条件
        self.enable_local_search = True  # 是否启用局部搜索
        # 局部搜索统计
        self.local_search_stats = {
            'calls': 0,
            'improvements': 0,
            'total_improvement': 0.0
        }
        # 局部搜索策略参数
        self.ls_trigger_strategies = {
            'improvement_based': True,  # 基于改进的触发
            'cost_threshold': True,  # 基于成本阈值的触发
            'frequency_based': True,  # 基于频率的触发
            'adaptive_threshold': True  # 自适应阈值
        }
        self.ls_call_frequency = 10  # 每10次操作强制调用一次
        self.ls_operation_count = 0  # 操作计数器
        self.ls_adaptive_theta = 0.05  # 自适应阈值初值
        print(" 局部搜索模块初始化完成")

        self.Tdis=None                                              #卡  车 —— 客户矩阵
        self.Ddis=None                                              #无 人 机 —— 客户距离矩阵
        self.Vist_T = None                                          #时间矩阵 ————记录车辆在客户处到达、离开的时间      包含所有节点的服务时间 包含起点 终点
        self.Distance()                                             #初始化距离矩阵
        self.T = np.empty((self.cnum, 2), dtype='object')     #时间矩阵 ————记录车辆到达客户的时间，主要用以时间顺序排列 用 object 类型初始化，避免重复赋值        仅包含客户的服务时间
        self.Initial_vehicle_information()
        self.Initial_visit_T()
        self.clean_route_data_types()
        # 初始化客户服务状态
        self._initialize_customer_success_status()

        self._initialize_vehicle_customer_assignment()
        ###### 构建初始解 ######
        # 逐个添加卡车
        for truck in self.TRUCK_Routes:
            try:
                truck_cost = self.cost_single_vehicle(truck.vehicle_id)
                self.Initial_solution.add_truck(truck, truck_cost)
                print(f"    添加卡车 {truck.vehicle_id}: 成本 {truck_cost:.2f}")
            except Exception as e:
                print(f"    添加卡车 {truck.vehicle_id} 失败: {e}")
        # 逐个添加无人机
        for drone in self.DRONE_Routes:
            try:
                drone_cost = 0  # 无人机成本已经包含在对应卡车成本中
                self.Initial_solution.add_drone(drone, drone_cost)
                print(f"    添加无人机 {drone.vehicle_id}: {len(drone.route)} 个任务")
            except Exception as e:
                print(f"    添加无人机 {drone.vehicle_id} 失败: {e}")
        # 逐个添加客户
        try:
            for customer in self.customers:
                self.Initial_solution.add_customer(customer.cust_no, customer)
            print(f"    添加客户: {len(self.customers)} 个")
        except Exception as e:
            print(f"    添加客户失败: {e}")
        # 计算并设置总成本
        try:
            total_cost = self.cost()
            self.Initial_solution.set_cost(total_cost)
            print(f"    初始解总成本: {total_cost:.2f}")
        except Exception as e:
            print(f"    计算总成本失败: {e}")
            total_cost = 0
            self.Initial_solution.set_cost(total_cost)
        print(" 初始解构建完成！")
        # 输出初始解信息
        self.Initial_solution.print_solution()
        print("\n 准备开始动态优化...")

    def _recalculate_trip_load_after_abandonment(self, trip: Dict, abandoned_customers: List[int]):
        """
        重新计算放弃客户后的trip载重
        使用适配器函数处理拼写不一致
        """
        try:
            # 送货载重保持起飞时的初始值（使用适配器函数）
            delivery_load = self._get_trip_initial_load(trip, 'delivery')

            # 取货载重只计算未被放弃的取货客户
            pickup_load = 0

            for customer_id in trip['path'][1:-1]:
                if customer_id <= len(self.customers):
                    customer = self.customers[customer_id - 1]
                    if customer.demand < 0 and customer_id not in abandoned_customers:
                        pickup_load += abs(customer.demand)

            # 更新载重信息
            new_total_load = delivery_load + pickup_load

            trip['current_load'] = new_total_load
            trip['current_load_delivery'] = delivery_load
            trip['current_load_pickup'] = pickup_load

            print(f"              载重更新: 送货{delivery_load}(固定), 取货{pickup_load}, 总计{new_total_load}")

        except Exception as e:
            print(f"              载重重新计算出错: {e}")

    def _initialize_vehicle_customer_assignment(self):
        """
        初始化每个车辆对负责的客户集合
        这个分配在车辆离开仓库后就不能改变（包裹不能转运约束）
        """
        print("\n初始化车辆对客户分配约束...")
        print("重要：一旦车辆离开仓库，包裹不能在车辆对间转运")
        for truck_id in range(len(self.TRUCK_Routes)):
            assigned_customers = set()
            delivery_packages = set()  # 该车辆对携带的送货包裹
            pickup_packages = set()  # 该车辆对需要取回的包裹
            # 1. 添加卡车直接服务的客户
            truck_route = self.TRUCK_Routes[truck_id].Troute
            print(f"   分析车辆对{truck_id} - 卡车路径: {truck_route}")
            for customer_id in truck_route[1:-1]:  # 排除起终点
                assigned_customers.add(customer_id)
                self.customer_to_vehicle[customer_id] = truck_id
                # 分析包裹类型
                customer = self.customers[customer_id - 1]
                if customer.demand > 0:
                    delivery_packages.add(customer_id)
                    print(f"     - 客户{customer_id}: 送货包裹 (需求={customer.demand})")
                else:
                    pickup_packages.add(customer_id)
                    print(f"     - 客户{customer_id}: 取货包裹 (需求={customer.demand})")
            # 2. 添加无人机服务的客户
            if self.DRONE_Routes[truck_id].route:
                print(f"   分析车辆对{truck_id} - 无人机路径: {len(self.DRONE_Routes[truck_id].route)}个任务")
                for trip_idx, trip in enumerate(self.DRONE_Routes[truck_id].route):
                    print(f"     任务{trip_idx}: {trip['path']}")
                    for customer_id in trip['path'][1:-1]:  # 排除起终点
                        assigned_customers.add(customer_id)
                        self.customer_to_vehicle[customer_id] = truck_id
                        # 分析包裹类型
                        customer = self.customers[customer_id - 1]
                        if customer.demand > 0:
                            delivery_packages.add(customer_id)
                            print(f"       - 客户{customer_id}: 无人机送货 (需求={customer.demand})")
                        else:
                            pickup_packages.add(customer_id)
                            print(f"       - 客户{customer_id}: 无人机取货 (需求={customer.demand})")
            # 3. 保存车辆对分配
            self.vehicle_customer_assignment[truck_id] = assigned_customers
            self.vehicle_initial_packages[truck_id] = {
                'delivery': delivery_packages,
                'pickup': pickup_packages
            }
            # 4. 输出分配结果
            total_delivery = sum(self.customers[c - 1].demand for c in delivery_packages)
            total_pickup = sum(abs(self.customers[c - 1].demand) for c in pickup_packages)
            print(f"   车辆对{truck_id}最终分配:")
            print(f"     - 负责客户: {sorted(assigned_customers)} (共{len(assigned_customers)}个)")
            print(f"     - 送货包裹: {len(delivery_packages)}个 (总重{total_delivery})")
            print(f"     - 取货包裹: {len(pickup_packages)}个 (总重{total_pickup})")
        # 5. 验证分配的完整性
        total_assigned = sum(len(customers) for customers in self.vehicle_customer_assignment.values())
        print(f"\n 分配验证:")
        print(f"   总客户数: {len(self.customers)}")
        print(f"   已分配客户数: {total_assigned}")
        print(f"   未分配客户数: {len(self.customers) - total_assigned}")
        if total_assigned != len(self.customers):
            print("发现未分配的客户，正在检查...")
            assigned_set = set()
            for customers in self.vehicle_customer_assignment.values():
                assigned_set.update(customers)
            unassigned = []
            for customer in self.customers:
                if customer.cust_no not in assigned_set:
                    unassigned.append(customer.cust_no)
            if unassigned:
                print(f" 未分配客户: {unassigned}")
            else:
                print(" 所有客户都已正确分配")
        else:
            print(" 所有客户都已正确分配")
        print(f"车辆对客户分配完成！")
        print("约束确立：动态优化过程中包裹不能在车辆对间转运\n")

    def validate_customer_assignment(self, truck_id: int, customer_id: int) -> bool:
        """
        验证客户是否属于指定车辆对
        """
        if truck_id not in self.vehicle_customer_assignment:
            return False
        return customer_id in self.vehicle_customer_assignment[truck_id]

    def get_vehicle_customers(self, truck_id: int) -> set:
        """
        获取指定车辆对负责的所有客户
        """
        return self.vehicle_customer_assignment.get(truck_id, set())

    def get_customer_vehicle(self, customer_id: int) -> int:
        """
        获取客户所属的车辆对ID
        """
        return self.customer_to_vehicle.get(customer_id, -1)

    def update_route(self, vex, customer_num, service_by):
        """
        路径更新方法
        """
        truck_id = service_by[1]
        # 关键验证：确保只处理指定车辆对的客户
        if not self.validate_customer_assignment(truck_id, customer_num):
            print(f" 约束违反：客户{customer_num}不属于车辆对{truck_id}")
            print(f"   客户{customer_num}实际属于车辆对: {self.get_customer_vehicle(customer_num)}")
            return
        print(f" 开始车辆对{truck_id}的约束版路径更新（客户{customer_num}）...")
        # 记录更新前的成本
        old_cost = self.cost()
        # 保留现有的时间更新逻辑
        launch_node = []
        retrieval_node = []
        if self.DRONE_Routes[truck_id].route:
            launch_node = [trip['launch_node'] for trip in self.DRONE_Routes[truck_id].route]
            retrieval_node = [trip['retrieval_node'] for trip in self.DRONE_Routes[truck_id].route]
        if service_by[0] == "tk":
            i_index = self.TRUCK_Routes[truck_id].Troute.index(customer_num)
            # 开始更新 Vist_T 时间
            self.Vist_T[customer_num - 1][2] = max(self.Vist_T[customer_num - 1][1],
                                                   self.customers[customer_num - 1].start_time)
            if customer_num not in launch_node and customer_num not in retrieval_node:
                self.Vist_T[customer_num - 1][3] = self.Vist_T[customer_num - 1][1]
                self.Vist_T[customer_num - 1][4] = self.Vist_T[customer_num - 1][2]
            if customer_num in launch_node and customer_num in retrieval_node:
                max_time = max(self.Vist_T[customer_num - 1][2], self.Vist_T[customer_num - 1][3])
                self.Vist_T[customer_num - 1][2] = max_time
                self.Vist_T[customer_num - 1][4] = self.Vist_T[customer_num - 1][3]
                for trip in self.DRONE_Routes[truck_id].route:
                    if trip['launch_node'] == customer_num:
                        path = trip['path']
                        for i in range(1, len(path)):
                            prev_indices = path[i - 1] - 1
                            current_indices = path[i] - 1
                            distance = self.Ddis[prev_indices + 1][current_indices + 1]
                            self.Vist_T[current_indices][3] = self.Vist_T[i - 1][4] + distance / self.drone_speed
                            self.Vist_T[current_indices][4] = max(self.Vist_T[current_indices][3], self.customers[
                                self.Vist_T[current_indices][0] - 1].start_time) + self.service_time
                            if path[i] not in retrieval_node:
                                self.Vist_T[current_indices][1] = 0
                                self.Vist_T[current_indices][2] = 0
            if customer_num not in launch_node and customer_num in retrieval_node:
                max_time = max(self.Vist_T[customer_num - 1][2], self.Vist_T[customer_num - 1][3])
                self.Vist_T[customer_num - 1][2] = max_time
                self.Vist_T[customer_num - 1][4] = self.Vist_T[customer_num - 1][2]
            self.Update_visit_T(truck_id, i_index + 1)
        else:
            self.Vist_T[customer_num - 1][4] = self.Vist_T[customer_num - 1][3]
            for trip in self.DRONE_Routes[truck_id].route:
                if customer_num in trip['path']:
                    path = trip['path']
                    i_index = path.index(customer_num)
                    for i in range(i_index + 1, len(path)):
                        prev_indices = path[i - 1] - 1
                        current_indices = path[i] - 1
                        distance = self.Ddis[prev_indices + 1][current_indices + 1]
                        self.Vist_T[current_indices][3] = self.Vist_T[i - 1][4] + distance / self.drone_speed
                        self.Vist_T[current_indices][4] = max(self.Vist_T[current_indices][3], self.customers[
                            self.Vist_T[current_indices][0] - 1].start_time) + self.service_time
                        if path[i] not in retrieval_node:
                            self.Vist_T[current_indices][1] = 0
                            self.Vist_T[current_indices][2] = 0
                        else:
                            self.Vist_T[current_indices][2] = max(self.Vist_T[current_indices][2],
                                                                  self.Vist_T[current_indices][4])
                    index = self.TRUCK_Routes[truck_id].Troute.index(path[len(path) - 1])
                    self.Update_visit_T(truck_id, index + 1)
        print(" 删除客户前的路径状态:")
        print("   卡车路径：")
        print(f"     路径: {self.TRUCK_Routes[truck_id].Troute}")
        print("   无人机路径：")
        for trip in self.DRONE_Routes[truck_id].route:
            path = [int(x) for x in trip['path']]
            energy = trip['energy']
            delivery_total = sum(self.customers[c - 1].demand for c in path[1:-1] if self.customers[c - 1].demand > 0)
            pickup_total = sum(self.customers[c - 1].demand for c in path[1:-1] if self.customers[c - 1].demand < 0)
            print(f"     路径: {path}, 总派送需求: {delivery_total}, 总取件需求: {pickup_total}, 总耗能: {energy}")
        if vex == 1:
            # 执行新的约束感知重优化流程
            reopt_success, cost_improvement = self._execute_constraint_aware_reoptimization(
                truck_id, customer_num, old_cost)
            if reopt_success:
                new_cost = self.cost()
                total_improvement = old_cost - new_cost
                print(f"车辆对{truck_id}重优化成功！")
                print(f"  成本变化: {old_cost:.2f} → {new_cost:.2f}")
                print(f"  总改进: {total_improvement:.2f}")

                # 更新信息素（基于改进效果）
                if total_improvement > 0:
                    self.update_pheromone(old_cost, new_cost)
            else:
                print(f"车辆对{truck_id}重优化未能改进解")
        # 信息素更新和挥发
        self._pheromone_update_counter += 1
        if self._pheromone_update_counter % 10 == 0:
            self.evaporate_pheromone()
            if self._pheromone_update_counter % 50 == 0:  # 减少输出频率
                self.print_pheromone_info()

    def _execute_constraint_aware_reoptimization(self, truck_id, customer_num, old_cost):
        """
        新的约束感知重优化流程
        替代原来的 ALNS + 可行性修复 模式
        """
        print(f"  执行约束感知重优化...")

        try:
            # 1. 智能破坏：基于约束分析确定删除范围
            delete_list = self._intelligent_destroy_with_constraints(truck_id, customer_num)

            if not delete_list:
                print(f"    没有需要重新安排的客户")
                return True, False

            print(f"    智能删除{len(delete_list)}个客户: {delete_list}")

            # 2. 关键修改：约束感知修复（在修复过程中就保证可行性）
            repair_success = self._constraint_aware_repair(truck_id, delete_list)

            if not repair_success:
                print(f"    约束感知修复失败，启动应急修复...")
                repair_success = self.repair_ops.emergency_repair(truck_id, delete_list)

            # 3. 轻量级验证：只检查可能的遗漏问题
            if repair_success:
                minor_issues = self._lightweight_constraint_check(truck_id)
                if minor_issues:
                    print(f"    发现轻微问题，进行针对性修复...")
                    self._fix_minor_constraint_issues(truck_id, minor_issues)

            # 4. 局部搜索（条件触发）
            new_cost = self.cost()
            if repair_success and new_cost < old_cost * (1 + self.theta):
                print(f"    解质量良好，启动局部搜索...")
                optimized_cost = self.local_search(truck_id, new_cost)

            final_cost = self.cost()
            total_improvement = old_cost - final_cost

            return repair_success, total_improvement > 0

        except Exception as e:
            print(f"    约束感知重优化出错: {e}")
            return False, False

    def _constraint_aware_repair(self, truck_id, customers_to_repair):
        """
        约束感知的修复算子 - 替代原来的repair_ops.repair_solution
        关键区别：在插入时就保证约束满足，而不是插入后再修复
        """
        if not customers_to_repair:
            return True

        print(f"    开始约束感知修复，处理{len(customers_to_repair)}个客户")

        # 按约束友好度排序客户（载重小的优先）
        sorted_customers = sorted(customers_to_repair,
                                  key=lambda c: abs(self.customers[c - 1].demand))

        inserted_count = 0
        for customer_id in sorted_customers:
            # 关键：尝试约束兼容插入
            if self._constraint_compatible_insertion(truck_id, customer_id):
                inserted_count += 1
                print(f"        客户{customer_id}约束兼容插入成功")
            else:
                print(f"        客户{customer_id}约束兼容插入失败")

        success_rate = inserted_count / len(customers_to_repair)
        print(f"    约束感知修复完成: {inserted_count}/{len(customers_to_repair)} ({success_rate:.1%})")

        return inserted_count == len(customers_to_repair)

    def _constraint_compatible_insertion(self, truck_id, customer_id):
        """
        约束兼容的客户插入 - 核心的新逻辑
        """
        customer = self.customers[customer_id - 1]

        # 1. 优先尝试卡车插入（检查载重约束）
        truck_positions = self._get_load_safe_truck_positions(truck_id, customer_id)
        if truck_positions:
            # 选择成本最低的安全位置
            best_position = min(truck_positions, key=lambda p: p['cost'])
            if self._execute_load_safe_insertion(truck_id, customer_id, best_position['position']):
                return True

        # 2. 如果卡车不可行，尝试无人机
        if (customer.drone_eligible == 1 and
                abs(customer.demand) <= self.drone_max_capacity):

            drone_positions = self._get_constraint_safe_drone_positions(truck_id, customer_id)
            if drone_positions:
                best_position = min(drone_positions, key=lambda p: p['cost'])
                if self._execute_drone_constraint_safe_insertion(truck_id, customer_id, best_position):
                    return True

        return False

    def _get_load_safe_truck_positions(self, truck_id, customer_id):
        """
        获取载重安全的卡车插入位置
        这是约束感知的核心：插入前就检查安全性
        """
        safe_positions = []
        truck_route = self.TRUCK_Routes[truck_id].Troute
        customer = self.customers[customer_id - 1]

        # 检查每个可能的插入位置
        for pos in range(1, len(truck_route)):
            # 模拟在此位置插入后的载重安全性
            if self._simulate_insertion_safety(truck_id, customer_id, pos):
                cost = self._calculate_insertion_cost(truck_id, customer_id, pos, 'truck')
                safe_positions.append({
                    'position': pos,
                    'cost': cost
                })

        return safe_positions

    def _simulate_insertion_safety(self, truck_id, customer_id, position):
        """
        模拟插入的安全性检查
        """
        customer = self.customers[customer_id - 1]
        truck = self.TRUCK_Routes[truck_id]

        # 从当前载重开始模拟
        simulated_load = truck.current_load
        truck_route = truck.Troute

        # 在插入位置处理新客户
        if customer.demand < 0:  # 取货
            simulated_load += abs(customer.demand)
            if simulated_load > truck.max_capacity:
                return False  # 立即超载
        else:  # 送货
            simulated_load -= customer.demand

        # 检查对后续客户的影响
        for i in range(position, len(truck_route) - 1):
            future_customer_id = truck_route[i]
            if future_customer_id <= len(self.customers):
                future_customer = self.customers[future_customer_id - 1]

                if future_customer.success is None:  # 未服务的客户
                    if future_customer.demand < 0:  # 取货
                        simulated_load += abs(future_customer.demand)
                        if simulated_load > truck.max_capacity:
                            return False  # 会导致后续超载
                    else:  # 送货
                        simulated_load -= future_customer.demand

        return True  # 安全

    def _lightweight_constraint_check(self, truck_id):
        """
        轻量级约束检查 - 替代完整的可行性检查
        只检查可能遗漏的关键问题
        """
        issues = []

        # 快速检查卡车当前载重
        truck = self.TRUCK_Routes[truck_id]
        if truck.current_load > truck.max_capacity:
            issues.append({
                'type': 'truck_overload',
                'severity': 'high',
                'excess': truck.current_load - truck.max_capacity
            })

        # 快速检查无人机载重和能耗
        for trip_idx, trip in enumerate(self.DRONE_Routes[truck_id].route):
            if trip.get('current_load', 0) > self.drone_max_capacity:
                issues.append({
                    'type': 'drone_overload',
                    'trip_index': trip_idx,
                    'severity': 'high'
                })

            if trip.get('energy', 0) > self.drone_max_battery:
                issues.append({
                    'type': 'drone_energy_exceeded',
                    'trip_index': trip_idx,
                    'severity': 'critical'
                })

        return issues

    def _check_forward_looking_constraints_after_failure(self, vehicle_id: int, failed_customer_id: int) -> Dict:
        """
        服务失败后的前瞻性约束检查
        检查按原计划继续执行时，后续约束是否能满足
        Args:
            vehicle_id: 车辆对ID
            failed_customer_id: 服务失败的客户ID

        Returns:
            包含所有约束违反信息的字典
        """
        print(f"     执行车辆对{vehicle_id}前瞻性约束检查（客户{failed_customer_id}服务失败）...")
        violations = {
            'truck_load': [],
            'drone_load': [],
            'drone_energy': [],
            'any_violation': False
        }

        try:
            # 1. 检查卡车载重前瞻性约束
            truck_violations = self._check_truck_forward_looking_load(vehicle_id, failed_customer_id)
            if truck_violations:
                violations['truck_load'] = truck_violations
                violations['any_violation'] = True
                print(f"       发现{len(truck_violations)}个卡车载重约束违反")
            # 2. 检查无人机载重前瞻性约束
            drone_load_violations = self._check_drone_forward_looking_load(vehicle_id, failed_customer_id)
            if drone_load_violations:
                violations['drone_load'] = drone_load_violations
                violations['any_violation'] = True
                print(f"       发现{len(drone_load_violations)}个无人机载重约束违反")
            # 3. 检查无人机能耗前瞻性约束
            drone_energy_violations = self._check_drone_forward_looking_energy(vehicle_id, failed_customer_id)
            if drone_energy_violations:
                violations['drone_energy'] = drone_energy_violations
                violations['any_violation'] = True
                print(f"       发现{len(drone_energy_violations)}个无人机能耗约束违反")

            if not violations['any_violation']:
                print(f"       所有前瞻性约束都能满足，无需重优化")
        except Exception as e:
            print(f"       前瞻性约束检查出错: {e}")
            violations['any_violation'] = True  # 出错时保守地触发重优化

        return violations

    def _check_truck_forward_looking_load(self, vehicle_id: int, failed_customer_id: int) -> List[Dict]:
        """
        检查卡车按原计划继续执行时的载重约束
        """
        violations = []
        try:
            truck_route = self.TRUCK_Routes[vehicle_id].Troute
            failed_customer = self.customers[failed_customer_id - 1]

            # 找到失败客户在路径中的位置
            if failed_customer_id not in truck_route:
                return violations

            failed_position = truck_route.index(failed_customer_id)

            # 计算失败客户之前的载重状态
            current_load = self.TRUCK_Routes[vehicle_id].initial_load

            # 模拟到失败客户之前的载重变化
            for i in range(1, failed_position):
                customer_id = truck_route[i]
                if customer_id <= len(self.customers):
                    customer = self.customers[customer_id - 1]
                    if customer.success is True:
                        current_load -= customer.demand
                    elif customer.success is False:
                        # 之前失败的客户：送货失败包裹仍在车上，取货失败没取到货
                        if customer.demand > 0:  # 送货失败
                            pass  # 载重不变
                        else:  # 取货失败
                            pass  # 载重不变
                    else:
                        # 未服务客户，假设按计划执行
                        current_load -= customer.demand

            # 处理当前失败客户对载重的影响
            if failed_customer.demand > 0:
                # 送货失败：包裹仍在车上，载重不减少
                pass
            else:
                # 取货失败：没取到货，载重不增加
                pass

                # 检查后续客户按原计划执行时是否会违反约束
            for i in range(failed_position + 1, len(truck_route) - 1):
                customer_id = truck_route[i]
                if customer_id <= len(self.customers):
                    customer = self.customers[customer_id - 1]

                    if customer.success is True:
                        # 已成功服务
                        current_load -= customer.demand
                    elif customer.success is False:
                        # 已失败服务
                        if customer.demand > 0:
                            pass  # 送货失败，载重不变
                        else:
                            pass  # 取货失败，载重不变
                    else:
                        # 未服务客户，检查按计划执行是否违反约束
                        if customer.demand < 0:  # 取货客户
                            projected_load = current_load + abs(customer.demand)
                            if projected_load > self.truck_max_capacity:
                                violations.append({
                                    'customer_id': customer_id,
                                    'position': i,
                                    'violation_type': 'future_pickup_overload',
                                    'current_load': current_load,
                                    'pickup_demand': abs(customer.demand),
                                    'projected_load': projected_load,
                                    'excess': projected_load - self.truck_max_capacity,
                                    'caused_by_failure': failed_customer_id
                                })

                        # 按计划更新载重
                        current_load -= customer.demand

        except Exception as e:
            print(f"         卡车前瞻性载重检查出错: {e}")

        return violations

    def _check_drone_forward_looking_load(self, vehicle_id: int, failed_customer_id: int) -> List[Dict]:
        """
        检查无人机按原计划继续执行时的载重约束
        """
        violations = []
        try:
            for trip_idx, trip in enumerate(self.DRONE_Routes[vehicle_id].route):
                path = trip['path']

                # 检查失败客户是否在这个无人机路径中
                if failed_customer_id not in path:
                    continue

                failed_position = path.index(failed_customer_id)
                failed_customer = self.customers[failed_customer_id - 1]

                # 计算到失败客户之前的载重
                current_load = trip['initial_load']
                for i in range(1, failed_position):
                    customer_id = path[i]
                    if customer_id <= len(self.customers):
                        customer = self.customers[customer_id - 1]
                        if customer.success is not False:  # 成功或未服务
                            current_load -= customer.demand

                # 处理失败客户对载重的影响
                if failed_customer.demand > 0:
                    # 送货失败：包裹仍在无人机上
                    pass
                else:
                    # 取货失败：没取到货，载重不增加
                    pass

                # 检查后续客户
                for i in range(failed_position + 1, len(path) - 1):
                    customer_id = path[i]
                    if customer_id <= len(self.customers):
                        customer = self.customers[customer_id - 1]

                        if customer.success is None:  # 未服务客户
                            if customer.demand < 0:  # 取货客户
                                projected_load = current_load + abs(customer.demand)
                                if projected_load > self.drone_max_capacity:
                                    violations.append({
                                        'trip_index': trip_idx,
                                        'customer_id': customer_id,
                                        'position': i,
                                        'violation_type': 'drone_future_pickup_overload',
                                        'current_load': current_load,
                                        'pickup_demand': abs(customer.demand),
                                        'projected_load': projected_load,
                                        'excess': projected_load - self.drone_max_capacity,
                                        'caused_by_failure': failed_customer_id
                                    })

                        # 按计划更新载重
                        if customer.success is not False:
                            current_load -= customer.demand

        except Exception as e:
            print(f"         无人机前瞻性载重检查出错: {e}")

        return violations

    def _check_drone_forward_looking_energy(self, vehicle_id: int, failed_customer_id: int) -> List[Dict]:
        """
        检查无人机按原计划继续执行时的能耗约束
        关键：检查无人机是否能支持返回卡车
        """
        violations = []
        try:
            for trip_idx, trip in enumerate(self.DRONE_Routes[vehicle_id].route):
                path = trip['path']

                # 检查失败客户是否在这个无人机路径中
                if failed_customer_id not in path:
                    continue

                failed_position = path.index(failed_customer_id)

                # 重新计算从失败点开始到返回的能耗
                remaining_energy = self._calculate_remaining_energy_after_failure(
                    trip, failed_position, failed_customer_id)

                if remaining_energy > self.drone_max_battery:
                    violations.append({
                        'trip_index': trip_idx,
                        'violation_type': 'insufficient_energy_to_return',
                        'failed_customer': failed_customer_id,
                        'failed_position': failed_position,
                        'required_energy': remaining_energy,
                        'available_energy': self.drone_max_battery,
                        'energy_deficit': remaining_energy - self.drone_max_battery
                    })

        except Exception as e:
            print(f"         无人机前瞻性能耗检查出错: {e}")

        return violations

    def _calculate_remaining_energy_after_failure(self, trip: Dict, failed_position: int,
                                                  failed_customer_id: int) -> float:
        """计算从失败客户位置继续执行剩余路径所需的能耗"""
        try:
            path = trip['path']
            failed_customer = self.customers[failed_customer_id - 1]  # 正确定义失败客户

            # 计算失败客户处的载重状态
            current_load = trip['current_load']

            # 从失败客户开始到路径结束的剩余能耗计算
            total_remaining_energy = 0

            for i in range(failed_position, len(path) - 1):
                from_customer_id = path[i]
                to_customer_id = path[i + 1]

                # 飞行能耗
                distance = self.ALLdistanceDmatrix[from_customer_id][to_customer_id]
                flight_time = distance / self.drone_speed
                flight_energy = flight_time * self.energy_fight * (current_load + self.drone_weight)
                total_remaining_energy += flight_energy

                # 如果不是最后一个客户（即不是回收节点）
                if i < len(path) - 2:
                    to_customer = self.customers[to_customer_id - 1]  # 正确获取客户对象

                    # 等待能耗
                    if hasattr(to_customer, 'arrive_drone') and hasattr(to_customer, 'start_time'):
                        arrive_time = getattr(to_customer, 'arrive_drone', to_customer.start_time)
                        if arrive_time < to_customer.start_time:
                            wait_time = to_customer.start_time - arrive_time
                            hover_energy = wait_time * self.energy_hover * (current_load + self.drone_weight)
                            total_remaining_energy += hover_energy

                    # 服务能耗
                    service_energy = self.service_time * self.energy_service * (current_load + self.drone_weight)
                    total_remaining_energy += service_energy

                    # 更新载重（修复原错误）
                    if to_customer.success is None:  # 未服务客户
                        if to_customer.demand > 0:  # 修正：使用 to_customer 而非未定义的 customer
                            current_load -= to_customer.demand
                        else:
                            current_load += abs(to_customer.demand)

            return total_remaining_energy

        except Exception as e:
            print(f"计算剩余能耗失败: {e}")
            return self.drone_max_battery * 1.1  # 保守估计

    def run_dynamic_optimization(self):
        """
        多阶段动态规划主函数
        """
        try:
            # 步骤1: 初始化时间矩阵T（按服务时间排序所有客户）
            self._initialize_time_matrix_T()
            total_stages = len(self.T)
            print(f" 总阶段数: {total_stages}")
            # 步骤2: 多阶段处理主循环
            current_stage_index = 0
            processed_customers = set()
            while current_stage_index < len(self.T):
                # 获取当前阶段的客户
                current_customer_id = int(self.T[current_stage_index][0])
                current_service_time = self.T[current_stage_index][1]
                print(f"\n 阶段 {current_stage_index + 1}: 客户{current_customer_id}")
                print(f"    预计服务时间: {current_service_time:.2f}")
                # 跳过已处理的客户
                if current_customer_id in processed_customers:
                    print(f"    客户{current_customer_id}已处理，跳过")
                    current_stage_index += 1
                    continue
                # 获取客户信息
                customer = self.customers[current_customer_id - 1]
          # ================== 阶段1：服务前准备和状态更新 ==================
                # 无人机客户：到达时立即更新电池状态（无论后续服务成功与否）
                if (hasattr(customer, 'service_by') and customer.service_by and
                        customer.service_by[0] == "de"):
                    print(f"    客户{current_customer_id}由无人机服务，更新电池状态")
                    self._update_drone_battery_on_arrival(current_customer_id)

                at_home_probability = getattr(customer, 'possibility', 0.8)
                # 核心决策：判断客户是否在家
                random_number = random.random()
                if random_number <= at_home_probability:
                    # ============ 服务成功分支 ============
                    print(f"    客户{current_customer_id}在家，服务成功")
                    success_result = self._handle_successful_service_complete(current_customer_id)
                    processed_customers.add(current_customer_id)
                    current_stage_index += 1
                else:
                    # ============ 服务失败分支 ============
                    print(f"    客户{current_customer_id}不在家，服务失败")
                    # 阶段1：服务失败状态更新
                    failure_result = self._handle_failed_service_complete(current_customer_id)
                    # 标记客户已处理（无论是否重规划）
                    processed_customers.add(current_customer_id)
        # ================== 阶段2：单车辆对约束检查==================
                    vehicle_id = self.get_customer_vehicle(current_customer_id)
                    constraint_analysis = self._comprehensive_constraint_analysis(
                        vehicle_id, current_customer_id, failure_result)
        # ================== 阶段3：差异化重规划决策==================
                    if constraint_analysis['requires_replanning']:
                        print(f"     需要重规划车辆对{vehicle_id}")
                        # 根据服务类型选择差异化策略
                        service_type = customer.service_by[0] if customer.service_by else None
                        if service_type == "de":  # 无人机服务失败
                            replan_success = self._drone_failure_specialized_replanning(
                                vehicle_id, current_customer_id, constraint_analysis)
                        elif service_type == "tk":  # 卡车服务失败
                            replan_success = self._truck_failure_specialized_replanning(
                                vehicle_id, current_customer_id, constraint_analysis)
                        else:
                            replan_success = False
                        # 重规划成功后更新时间矩阵
                        if replan_success:
                            print(f"     重规划成功，更新时间矩阵")
                            self._recalculate_time_matrix_T()
                            # 找到下一个未处理的客户
                            next_stage = self._find_next_unprocessed_stage(processed_customers)
                            current_stage_index = next_stage if next_stage is not None else len(self.T)
                        else:
                            print(f"     重规划失败，继续下一客户")
                            current_stage_index += 1
                    else:
                        print(f"     约束满足，无需重规划")
                        current_stage_index += 1
            print(f"\n🏁 改进版多阶段动态规划完成！")
            print(f"总服务客户数: {sum(1 for c in self.customers if c.success is True)}")
            print(f"总失败客户数: {sum(1 for c in self.customers if c.success is False)}")
            return True
        except Exception as e:
            print(f"❌ 动态规划执行出错: {e}")
            return False
#约束判断
    def _comprehensive_constraint_analysis(self, vehicle_id, failed_customer_id, failure_result):
        """
        阶段2：全面的约束分析
        Level 1: 直接约束检查
        Level 2: 连锁效应分析
        """
        print(f"        🔍 执行车辆对{vehicle_id}约束分析...")
        analysis_result = {
            'vehicle_id': vehicle_id,
            'failed_customer': failed_customer_id,
            'service_type': failure_result.get('service_type'),
            'direct_violations': {},
            'cascading_effects': {},
            'requires_replanning': False,
            'severity_level': 'none'
        }
        try:
            # Level 1: 直接约束检查
            print(f"          📋 Level 1: 直接约束检查")
            # 1.1 根据服务类型进行针对性检查
            if failure_result.get('service_type') == 'de':
                # 无人机约束检查
                energy_violations = self._check_drone_energy_direct_constraints(
                    vehicle_id, failed_customer_id)
                drone_load_violations = self._check_drone_load_direct_constraints(
                    vehicle_id, failed_customer_id)
                analysis_result['direct_violations'].update({
                    'drone_energy': energy_violations,
                    'drone_load': drone_load_violations
                })
            elif failure_result.get('service_type') == 'tk':
                # 卡车约束检查
                truck_load_violations = self._check_truck_load_direct_constraints(
                    vehicle_id, failed_customer_id)
                analysis_result['direct_violations'].update({
                    'truck_load': truck_load_violations
                })

            # Level 2: 连锁效应分析
            print(f"           Level 2: 连锁效应分析")
            cascading_effects = self._analyze_cascading_effects_within_vehicle(
                vehicle_id, failed_customer_id, analysis_result['direct_violations'])
            analysis_result['cascading_effects'] = cascading_effects

            # 综合判断是否需要重规划
            analysis_result['requires_replanning'] = (
                    self._has_critical_violations(analysis_result['direct_violations']) or
                    self._has_critical_cascading_effects(cascading_effects)
            )

            # 确定严重程度
            analysis_result['severity_level'] = self._determine_violation_severity(
                analysis_result['direct_violations'], cascading_effects)

            print(f"           约束分析完成，需要重规划: {analysis_result['requires_replanning']}")

            return analysis_result

        except Exception as e:
            print(f"          ❌ 约束分析出错: {e}")
            analysis_result['requires_replanning'] = True  # 保守策略
            return analysis_result

    #更新离开节点时的当前剩余电量
    def _update_drone_battery_on_leave(self, customer_id: int, success_status: bool):
        """
        无人机离开客户时更新电池状态
        此方法在判断服务成功/失败之前调用
        """
        try:
                # 在开头初始化能耗变量
                hover_energy = 0.0  # 添加这行
                service_energy = 0.0  # 添加这行
                target_customer = self.customers[customer_id - 1]
                vehicle_id = target_customer.service_by[1]
                print(f"       更新车辆对{vehicle_id}无人机剩余电池：离开客户{customer_id}")
                # 找到该客户所在的trip
                target_trip = None
                for  trip in self.DRONE_Routes[vehicle_id].route:
                    if customer_id in trip['path']:
                        target_trip = trip
                        break
                if target_trip is None:
                    print(f"         未找到客户{customer_id}所在的无人机trip")
                    return
                #  等待悬停能耗（如果早到）
                if hasattr(target_customer, 'arrive_drone') and hasattr(target_customer, 'start_time'):
                    arrive_time = getattr(target_customer, 'arrive_drone', target_customer.start_time)
                    if arrive_time < target_customer.start_time:
                        wait_time = target_customer.start_time - arrive_time
                        hover_energy = wait_time * self.energy_hover * (target_trip['current_load'] + self.drone_weight)
                        print(f"           等待悬停能耗: {hover_energy:.2f} (等待{wait_time:.2f}时间单位)")
                # 服务能耗
                service_energy = self.service_time * self.energy_service * (target_trip['current_load'] + self.drone_weight)
                energy_consumed=hover_energy+service_energy
                target_trip['current_remain_battery']=target_trip['current_remain_battery']-energy_consumed
        except Exception as e:
            print(f"         无人机电池更新出错: {e}")

    # 更新到达节点时的当前剩余电量
    def _update_drone_battery_on_arrival(self, customer_id: int):
        """
        无人机到达客户时更新电池状态
        此方法在判断服务成功/失败之前调用
        """
        try:
            customer = self.customers[customer_id - 1]
            vehicle_id = customer.service_by[1]
            print(f"       更新车辆对{vehicle_id}无人机电池：到达客户{customer_id}")
            # 找到该客户所在的trip
            target_trip = None
            target_trip_idx = None
            for trip_idx, trip in enumerate(self.DRONE_Routes[vehicle_id].route):
                if customer_id in trip['path']:
                    target_trip = trip
                    target_trip_idx = trip_idx
                    break
            if target_trip is None:
                print(f"         未找到客户{customer_id}所在的无人机trip")
                return

            path = target_trip['path']
            target_position = path.index(customer_id)
            if target_position == 0:  # 如果是起飞节点，不消耗能耗
                energy_consumed = 0
            # 获取前一个位置和当前位置
            from_customer_id = path[target_position - 1]
            to_customer_id = customer_id
            # 计算飞行能耗
            current_load = trip['current_load']
            distance = self.ALLdistanceDmatrix[from_customer_id][to_customer_id]
            flight_time = distance / self.drone_speed
            flight_energy = flight_time * self.energy_fight * (current_load + self.drone_weight)
            energy_consumed = flight_energy
            trip['current_remain_battery'] = trip['current_remain_battery'] - energy_consumed
        except Exception as e:
            print(f"         无人机电池更新出错: {e}")

    def _handle_successful_service_complete(self, customer_id):
        """
        处理成功服务的客户
        """
        print(f"     处理客户{customer_id}成功服务...")
        try:
            # 1. 设置客户服务状态
            self.set_customer_service_status(customer_id, True)
            #2. 如果是无人机服务客户，则需要在服务成功以后更新剩余电量
            # 获取客户信息
            customer = self.customers[customer_id - 1]
            if (hasattr(customer, 'service_by') and customer.service_by and
                    customer.service_by[0] == "de"):
                print(f"    客户{customer_id}由无人机服务，更新电池状态")
                self._update_drone_battery_on_leave(customer_id,True)
            # 3. 立即更新所有相关车辆的载重
            self._update_all_vehicle_loads_after_service(customer_id)
            print(f"       客户{customer_id}成功服务处理完成")
            return {'status': 'success', 'customer_id': customer_id}
        except Exception as e:
            print(f"       成功服务处理出错: {e}")
            return {'status': 'error', 'customer_id': customer_id, 'error': str(e)}

    def _handle_failed_service_complete(self, customer_id):
        print(f"     处理客户{customer_id}服务失败...")
        try:
            # 1. 设置客户服务状态
            self.set_customer_service_status(customer_id, False)
            #2. 如果是无人机服务客户，则需要更新以后更新剩余电量
            # 获取客户信息
            customer = self.customers[customer_id - 1]
            if (hasattr(customer, 'service_by') and customer.service_by and
                    customer.service_by[0] == "de"):
                print(f"    客户{customer_id}由无人机服务，更新电池状态")
                self._update_drone_battery_on_leave(customer_id,False)
            # 3. 立即更新所有相关车辆的载重
            self._update_all_vehicle_loads_after_service(customer_id)
            return {
                'status': 'failed',
                'customer_id': customer_id,
                'service_type': customer.service_by[0] if customer.service_by else None,
                'vehicle_id': self.get_customer_vehicle(customer_id)
            }
        except Exception as e:
            print(f"      ❌ 失败服务处理出错: {e}")
            return {'status': 'error', 'customer_id': customer_id, 'error': str(e)}

    def _check_drone_load_direct_constraints(self, vehicle_id: int, failed_customer_id: int) -> Dict:
        """
        检查无人机服务失败后的载重直接约束
        Args:
            vehicle_id: 车辆对ID
            failed_customer_id: 服务失败的客户ID
        Returns:
            包含载重约束违反信息的字典
        """
        violations = {
            'has_violations': False,
            'overload_points': [],
            'max_violation': 0.0,
            'affected_trip': None,
            'critical_customers': []
        }

        try:
            # 1. 找到失败客户所在的无人机trip
            failed_trip = None
            trip_index = None
            for idx, trip in enumerate(self.DRONE_Routes[vehicle_id].route):
                if failed_customer_id in trip['path']:
                    failed_trip = trip
                    trip_index = idx
                    break
            if failed_trip is None:
                # 失败客户不在无人机路径中，无需检查无人机载重
                return violations
            print(f"            找到失败客户{failed_customer_id}在无人机trip {trip_index}")
            # 2. 获取路径和失败位置
            path = failed_trip['path']
            failed_position = path.index(failed_customer_id)
            failed_customer = self.customers[failed_customer_id - 1]
            violations['affected_trip'] = trip_index

            # 3. 计算失败客户处的载重状态
            current_load = failed_trip['current_load']

            print(f"            失败客户处当前载重: {current_load:.2f}")

            # 4. 处理失败客户对载重的影响
            if failed_customer.demand > 0:
                # 送货失败：包裹仍在无人机上，载重不减少
                print(f"            送货客户失败，包裹{failed_customer.demand}仍在无人机上")
                # current_load 保持不变
            else:
                # 取货失败：没取到货，载重不增加
                print(f"            取货客户失败，未取到货物{abs(failed_customer.demand)}")
                # current_load 保持不变

            # 5. 模拟后续客户按原计划执行时的载重变化
            simulated_load = current_load
            for i in range(failed_position + 1, len(path) - 1):  # 排除回收节点
                customer_id = path[i]
                if customer_id <= len(self.customers):
                    future_customer = self.customers[customer_id - 1]
                    # 只检查未服务的客户（success=None）
                    if future_customer.success is None:
                        if future_customer.demand < 0:  # 取货客户
                            # 取货后载重增加
                            pickup_amount = abs(future_customer.demand)
                            projected_load = simulated_load + pickup_amount

                            # 检查是否超载
                            if projected_load > self.drone_max_capacity:
                                violation_info = {
                                    'customer_id': customer_id,
                                    'position_in_path': i,
                                    'operation_type': 'pickup',
                                    'pickup_amount': pickup_amount,
                                    'load_before': simulated_load,
                                    'projected_load': projected_load,
                                    'excess_load': projected_load - self.drone_max_capacity,
                                    'caused_by_failure': failed_customer_id
                                }
                                violations['overload_points'].append(violation_info)
                                violations['critical_customers'].append(customer_id)
                                violations['has_violations'] = True
                                # 更新最大违反量
                                if violation_info['excess_load'] > violations['max_violation']:
                                    violations['max_violation'] = violation_info['excess_load']

                                print(f"            检测到载重违反：客户{customer_id}取货后将达到{projected_load:.2f}")
                                print(f"              超载量: {violation_info['excess_load']:.2f}")
                            simulated_load = projected_load
                        else:  # 送货客户
                            # 送货后载重减少
                            simulated_load -= future_customer.demand
            return violations

        except Exception as e:
            print(f"            无人机载重约束检查出错: {e}")
            # 出错时保守地标记为有违反
            violations['has_violations'] = True
            violations['error'] = str(e)
            return violations

    def _check_drone_energy_direct_constraints(self, vehicle_id: int, failed_customer_id: int) -> Dict:
        """检查无人机服务失败后的能耗约束"""
        violations = {
            'has_violations': False,
            'energy_insufficient': None,
            'energy_warning': None,
            'severity': 'none'
        }

        try:
            # 找到失败客户所在的无人机trip
            failed_trip_idx = None
            failed_trip = None

            for trip_idx, trip in enumerate(self.DRONE_Routes[vehicle_id].route):
                if failed_customer_id in trip['path']:
                    failed_trip_idx = trip_idx
                    failed_trip = trip
                    break

            if failed_trip is None:
                return violations

            # 计算从失败客户位置继续执行到结束所需的能耗
            remaining_battery = failed_trip.get('current_remain_battery', self.drone_max_battery)
            path = failed_trip['path']
            failed_position = path.index(failed_customer_id)

            # 修正：正确计算剩余所需能耗
            remaining_energy_needed = self._calculate_remaining_energy_after_failure(
                failed_trip, failed_position, failed_customer_id)

            # 检查是否超出电池容量
            if remaining_energy_needed > remaining_battery:  # 修正：使用正确的变量名
                violations['has_violations'] = True
                violations['energy_insufficient'] = {
                    'trip_index': failed_trip_idx,
                    'failed_customer': failed_customer_id,
                    'remaining_battery': remaining_battery,
                    'required_energy': remaining_energy_needed,
                    'energy_deficit': remaining_energy_needed - remaining_battery,
                    'severity': 'critical'
                }
                violations['severity'] = 'critical'
                print(f"能耗不足: 缺口{remaining_energy_needed - remaining_battery:.2f}")
            else:
                # 检查能量裕度
                energy_margin = remaining_battery - remaining_energy_needed
                if energy_margin < self.drone_max_battery * 0.1:
                    violations['energy_warning'] = {
                        'trip_index': failed_trip_idx,
                        'energy_margin': energy_margin,
                        'severity': 'warning'
                    }
                    print(f"能量裕度不足: {energy_margin:.2f}")

        except Exception as e:
            print(f"无人机能耗约束检查出错: {e}")
            violations['has_violations'] = True
            violations['error'] = str(e)
            violations['severity'] = 'error'

        return violations

    def _calculate_remaining_energy_after_failure(self, trip: Dict, failed_position: int,
                                                  failed_customer_id: int) -> float:
        """
        计算从失败客户位置继续执行剩余路径所需的能耗
        考虑：失败客户包裹仍在无人机上（送货失败）或未取到（取货失败）
        """
        try:
            path = trip['path']
            failed_customer = self.customers[failed_customer_id - 1]
            # 计算失败客户处的载重状态
            current_load = trip['current_load']
            # 计算从失败客户开始到路径结束的剩余能耗
            energy_consumed = 0
            # 从失败客户到后续客户的能耗计算
            for i in range(failed_position, len(path) - 1):
                from_customer_id = path[i]
                to_customer_id = path[i + 1]
                # 飞行能耗
                distance = self.ALLdistanceDmatrix[from_customer_id][to_customer_id]
                flight_time = distance / self.drone_speed
                flight_energy = flight_time * self.energy_fight * (current_load + self.drone_weight)
                energy_consumed += flight_energy
                # 如果不是最后一个客户（即不是回收节点）
                if i < len(path) - 2:
                    to_customer = self.customers[to_customer_id - 1]
                    # 等待能耗（如果早到需要悬停）
                    if hasattr(to_customer, 'arrive_drone') and hasattr(to_customer, 'start_time'):
                        arrive_time = getattr(to_customer, 'arrive_drone', to_customer.start_time)
                        if arrive_time < to_customer.start_time:
                            wait_time = to_customer.start_time - arrive_time
                            hover_energy = wait_time * self.energy_hover * (current_load + self.drone_weight)
                            energy_consumed += hover_energy
                    # 服务能耗
                    service_energy = self.service_time * self.energy_service * (current_load + self.drone_weight)
                    energy_consumed += service_energy
                    # 更新载重（假设后续客户按计划成功服务）
                    if to_customer.success is None:  # 未服务客户
                        if to_customer.demand > 0:
                            current_load -= to_customer.demand
                        else:
                            current_load += abs(to_customer.demand)
            return energy_consumed
        except Exception as e:
            print(f"           计算剩余能耗失败: {e}")
            # 保守估计：返回最大能耗触发重规划
            return self.drone_max_battery * 1.1

    def _check_truck_load_direct_constraints(self, vehicle_id: int, failed_customer_id: int) -> Dict:
        """
        检查卡车服务失败后的载重约束
        关键:检查后续取货客户是否会导致超载，包括无人机起飞/回收的影响
        """
        violations = {}
        try:
            truck = self.TRUCK_Routes[vehicle_id]
            truck_route = truck.Troute
            if failed_customer_id not in truck_route:
                return violations

            failed_position = truck_route.index(failed_customer_id)
            failed_customer = self.customers[failed_customer_id - 1]
            print(f"         检查卡车载重约束(失败位置: {failed_position})")

            # 获取对应的无人机信息
            drone = self.DRONE_Routes[vehicle_id] if vehicle_id < len(self.DRONE_Routes) else None

            # 建立无人机节点映射
            drone_operations = {}
            if drone:
                for trip in drone.route:
                    launch_node = trip['launch_node']
                    retrieval_node = trip['retrieval_node']

                    drone_operations[launch_node] = {
                        'type': 'launch',
                        'trip': trip,
                        'total_initial_load': trip['initial_load']
                    }

                    drone_operations[retrieval_node] = {
                        'type': 'recovery',
                        'trip': trip,
                        'total_final_load': trip['current_load']
                    }

            # 获取当前载重
            current_load = truck.current_load
            print(f"         当前载重: {current_load:.2f} / {truck.max_capacity}")

            # 服务失败的影响处理
            if failed_customer.demand > 0:
                print(f"         送货失败,包裹{failed_customer.demand}仍在车上")
            else:
                print(f"         取货失败,未取到货物{abs(failed_customer.demand)}")

            # 检查后续路径中的所有操作
            future_overload_risks = []
            simulated_load = current_load

            for i in range(failed_position + 1, len(truck_route) - 1):
                node_id = truck_route[i]

                # 检查无人机操作节点
                if node_id in drone_operations:
                    operation = drone_operations[node_id]

                    if operation['type'] == 'launch':
                        # 无人机起飞：卡车载重减少
                        drone_initial_load = operation['total_initial_load']
                        simulated_load -= drone_initial_load
                        print(f"         无人机起飞节点{node_id}: 载重减少{drone_initial_load:.2f}")

                    elif operation['type'] == 'recovery':
                        # 无人机回收：卡车载重增加
                        drone_final_load = operation['total_final_load']
                        projected_load = simulated_load + drone_final_load

                        if projected_load > truck.max_capacity:
                            future_overload_risks.append({
                                'type': 'drone_recovery',
                                'node_id': node_id,
                                'position': i,
                                'weight_increase': drone_final_load,
                                'current_load': simulated_load,
                                'projected_load': projected_load,
                                'excess': projected_load - truck.max_capacity
                            })
                            print(f"         ⚠️ 无人机回收节点{node_id}将超载: "
                                  f"{projected_load:.2f} > {truck.max_capacity}")

                        simulated_load = projected_load
                        print(f"         无人机回收节点{node_id}: 载重增加{drone_final_load:.2f}")

                # 检查普通客户节点
                elif node_id <= len(self.customers):
                    future_customer = self.customers[node_id - 1]
                    if future_customer.success is None:
                        if future_customer.demand < 0:  # 取货客户
                            projected_load = simulated_load + abs(future_customer.demand)
                            if projected_load > truck.max_capacity:
                                future_overload_risks.append({
                                    'type': 'customer_pickup',
                                    'customer_id': node_id,
                                    'position': i,
                                    'pickup_demand': abs(future_customer.demand),
                                    'current_load': simulated_load,
                                    'projected_load': projected_load,
                                    'excess': projected_load - truck.max_capacity
                                })
                                print(f"         ⚠️ 客户{node_id}取货将超载: "
                                      f"{projected_load:.2f} > {truck.max_capacity}")
                            simulated_load = projected_load
                        else:  # 送货客户
                            simulated_load -= future_customer.demand

            # 记录违反
            if future_overload_risks:
                violations['truck_load_violations'] = {
                    'failed_customer': failed_customer_id,
                    'overload_risks': future_overload_risks,
                    'severity': 'critical',
                    'message': f'发现{len(future_overload_risks)}个潜在超载点'
                }

        except Exception as e:
            print(f"         卡车载重约束检查出错: {e}")
            violations['error'] = str(e)

        return violations

    def _handle_successful_service_complete(self, customer_id):
        """
        处理成功服务的客户
        """
        print(f"     处理客户{customer_id}成功服务...")
        try:
            # 1. 设置客户服务状态
            self.set_customer_service_status(customer_id, True)
            # 2. 如果是无人机服务客户，则需要在服务成功以后更新剩余电量
            # 获取客户信息
            customer = self.customers[customer_id - 1]
            if (hasattr(customer, 'service_by') and customer.service_by and
                    customer.service_by[0] == "de"):
                print(f"    客户{customer_id}由无人机服务，更新电池状态")
                self._update_drone_battery_on_leave(customer_id, True)
            # 3. 立即更新所有相关车辆的载重
            self._update_all_vehicle_loads_after_service(customer_id)
            print(f"       客户{customer_id}成功服务处理完成")
            return {'status': 'success', 'customer_id': customer_id}
        except Exception as e:
            print(f"      ❌ 成功服务处理出错: {e}")
            return {'status': 'error', 'customer_id': customer_id, 'error': str(e)}

    def _handle_failed_service_complete(self, customer_id):
        print(f"     处理客户{customer_id}服务失败...")
        try:
            # 1. 设置客户服务状态
            self.set_customer_service_status(customer_id, False)
            # 2. 如果是无人机服务客户，则需要在服务成功以后更新剩余电量
            # 获取客户信息
            customer = self.customers[customer_id - 1]
            if (hasattr(customer, 'service_by') and customer.service_by and
                    customer.service_by[0] == "de"):
                print(f"    客户{customer_id}由无人机服务，更新电池状态")
                self._update_drone_battery_on_leave(customer_id, False)
            # 3. 立即更新所有相关车辆的载重
            self._update_all_vehicle_loads_after_service(customer_id)
            return {
                'status': 'failed',
                'customer_id': customer_id,
                'service_type': customer.service_by[0] if customer.service_by else None,
                'vehicle_id': self.get_customer_vehicle(customer_id)
            }
        except Exception as e:
            print(f"      ❌ 失败服务处理出错: {e}")
            return {'status': 'error', 'customer_id': customer_id, 'error': str(e)}

    def _update_all_vehicle_loads_after_service(self, customer_id: int):
        """
        统一的载重更新方法 - 每次服务后调用
        Args:
            customer_id: 服务的客户ID
        """
        # 1. 确定客户所属的车辆对
        vehicle_id = self.get_customer_vehicle(customer_id)
        if vehicle_id < 0:
            print(f"         客户{customer_id}未分配到任何车辆对")
            return

        # 2. 从客户状态中获取服务结果
        customer = self.customers[customer_id - 1]
        success_status = customer.success  # True=成功, False=失败, None=待服务

        if success_status is True:
            status_text = "成功"
        elif success_status is False:
            status_text = "失败"
        else:
            status_text = "待服务"

        print(f"         更新车辆对{vehicle_id}载重状态（客户{customer_id}服务{status_text}）")

        # 3. 重新计算该车辆对的所有载重
        self._recalculate_vehicle_pair_loads(vehicle_id)

#服务成功后更新当前载重
    def _recalculate_vehicle_pair_loads(self, vehicle_id: int):
        """
        重新计算整个车辆对的载重状态
        基于所有客户的当前服务状态进行完整重算
        """
        print(f"         完整重算车辆对{vehicle_id}载重...")
        try:
            truck = self.TRUCK_Routes[vehicle_id]
            # 1. 重新计算卡车载重
            self._recalculate_truck_current_load(vehicle_id)
            # 2. 重新计算所有无人机任务载重
            self._recalculate_all_drone_trips_load(vehicle_id)
        except IndexError:
            raise Exception(f"车辆ID {vehicle_id} 不存在")
        except AttributeError as e:
            raise Exception(f"卡车对象缺少属性: {e}")
        except Exception as e:
            raise Exception(f"卡车载重计算失败: {e}")

    def _recalculate_truck_current_load(self, vehicle_id: int):
        """
        重新计算卡车当前载重
        """
        try:
            truck = self.TRUCK_Routes[vehicle_id]

            # 检查并初始化缺失的属性
            required_attrs = {
                'initial_load': 0,
                'initial_load_delivery': 0,
                'initial_load_pickup': 0,
                'current_load': 0,
                'current_load_delivery': 0,
                'current_load_pickup': 0
            }

            for attr_name, default_value in required_attrs.items():
                if not hasattr(truck, attr_name):
                    setattr(truck, attr_name, default_value)
                    print(f"           初始化 truck.{attr_name} = {default_value}")

            # 从初始载重开始重算
            truck.current_load = truck.initial_load
            truck.current_load_delivery = truck.initial_load_delivery
            truck.current_load_pickup = 0

            print(f"           卡车初始载重: {truck.initial_load}")

            # 根据卡车直接服务的客户状态调整载重
            for customer_id in truck.Troute[1:-1]:  # 排除起终点
                if customer_id <= len(self.customers):
                    customer = self.customers[customer_id - 1]
                    if customer.success is True:
                        # 已成功服务的客户
                        if customer.demand > 0:
                            # 成功送货：载重减少
                            truck.current_load -= customer.demand
                            truck.current_load_delivery -= customer.demand
                            print(f"           客户{customer_id}成功送货: -{customer.demand}")
                        else:
                            # 成功取货：载重增加
                            truck.current_load += abs(customer.demand)
                            truck.current_load_pickup += abs(customer.demand)
                            print(f"           客户{customer_id}成功取货: +{abs(customer.demand)}")
                    elif customer.success is False:
                        # 服务失败的客户
                        if customer.demand > 0:
                            # 送货失败：包裹仍在车上，载重不变
                            print(f"           客户{customer_id}送货失败: 包裹仍在车上")
                        else:
                            # 取货失败：没取到货，载重不变
                            print(f"           客户{customer_id}取货失败: 未取到货物")
                    # success is None：待服务，载重按初始状态

            # 处理无人机起飞/回收对卡车载重的影响
            self._adjust_truck_load_for_drone_operations(vehicle_id)

            print(f"           卡车当前载重: {truck.current_load}")

        except Exception as e:
            raise Exception(f"卡车载重计算失败: {e}")

    def _adjust_truck_load_for_drone_operations(self, vehicle_id: int):
        """
        调整卡车载重以考虑无人机操作的影响
        """
        try:
            truck = self.TRUCK_Routes[vehicle_id]
            # 确保属性存在
            if not hasattr(truck, 'current_load_delivery'):
                truck.current_load_delivery = truck.current_load
                # 获取无人机起飞和回收节点
                launch_nodes = []
                retrieval_nodes = []
                for trip in self.DRONE_Routes[vehicle_id].route:
                    launch_nodes.append(trip['launch_node'])
                    retrieval_nodes.append(trip['retrieval_node'])
                # 处理每个无人机任务对卡车载重的影响
                for trip in self.DRONE_Routes[vehicle_id].route:
                    launch_customer = self.customers[trip['launch_node'] - 1]
                    retrieval_customer = self.customers[trip['retrieval_node'] - 1]
                    # 起飞节点：如果卡车已到达且成功服务，无人机携带包裹起飞
                    if launch_customer.success is not None:
                        initial_drone_load = trip['initial_load']  # 无人机起飞时的包裹载重
                        truck.current_load -= initial_drone_load
                        truck.current_load_delivery -= trip['initial_load_delivery']
                        print(f"           无人机从客户{trip['launch_node']}起飞: 卡车载重减少{initial_drone_load}")
                    # 回收节点：如果卡车已到达且成功回收，无人机返回的包裹回到卡车
                    if retrieval_customer.success is not None:
                        actual_return_load = self._calculate_drone_actual_return_load(trip)
                        truck.current_load += actual_return_load
                        print(f"           无人机在客户{trip['retrieval_node']}回收: 卡车载重增加{actual_return_load}")
        except Exception as e:
            print(f"           调整卡车载重出错: {e}")

    def _calculate_drone_actual_return_load(self, trip):
        """
        计算无人机实际返回的载重
        基于无人机任务中每个客户的实际服务状态
        """
        return_load = 0
        return_delivery = 0  # 未送达的包裹
        return_pickup = 0  # 成功取到的货物

        for customer_id in trip['path'][1:-1]:  # 排除起终点
            if customer_id <= len(self.customers):
                customer = self.customers[customer_id - 1]
                if customer.demand > 0:  # 送货任务
                    if customer.success is False:
                        # 送货失败：包裹仍在无人机上，需要带回
                        return_delivery += customer.demand
                    # 送货成功：包裹已交付，无需带回
                else:  # 取货任务
                    if customer.success is True:
                        # 取货成功：货物在无人机上，需要带回
                        return_pickup += abs(customer.demand)
                    # 取货失败：没取到货，无需带回
        return_load = return_delivery + return_pickup
        return return_load

    def _recalculate_all_drone_trips_load(self, vehicle_id: int):
        """
        重新计算所有无人机任务的当前载重
        """
        for trip_idx, trip in enumerate(self.DRONE_Routes[vehicle_id].route):
            print(f"           重算无人机任务{trip_idx}载重...")
            # 重新计算这个trip的当前载重
            self._recalculate_single_trip_current_load(trip)
            # 重新计算能耗（基于当前载重）
            self._recalculate_trip_energy_with_current_status(trip)

    def _recalculate_single_trip_current_load(self, trip):
        """
        重新计算单个无人机任务的当前载重
        """
        # 当前载重从初始载重开始
        trip['current_load'] = trip['initial_load']
        trip['current_load_delivery'] = trip['initial_load_delivery']
        trip['current_load_pickup'] = 0
        # 根据路径中每个客户的服务状态调整
        for customer_id in trip['path'][1:-1]:  # 排除起终点
            if customer_id <= len(self.customers):
                customer = self.customers[customer_id - 1]
                if customer.success is True:
                    # 已成功服务
                    if customer.demand > 0:
                        # 成功送货：载重减少
                        trip['current_load'] -= customer.demand
                        trip['current_load_delivery'] -= customer.demand
                    else:
                        # 成功取货：载重增加
                        trip['current_load'] += abs(customer.demand)
                        trip['current_load_pickup'] += abs(customer.demand)
                # success is False 或 None：载重不变

    def _recalculate_trip_energy_with_current_status(self, trip):
        """
        基于当前服务状态重新计算任务能耗
        """
        if len(trip['path']) <= 2:
            trip['energy'] = 0
            return

        try:
            first_node_idx = trip['path'][0] - 1
            if first_node_idx >= 0:
                trip['energy'] = self.calculate_energy(
                    self.Vist_T[first_node_idx][4],
                    trip['path'],
                    trip['initial_load'],  # 起飞时的载重
                )
        except Exception as e:
            print(f"             任务能耗重算失败: {e}")
            # 保守估计
            trip['energy'] = self.drone_max_battery * 0.8

# ==================== 无人机重规划   ===================

    def _drone_failure_specialized_replanning(self, vehicle_id: int, failed_customer_id: int,
                                              constraint_analysis: Dict) -> bool:
        """
        阶段3：无人机服务失败专门重规划策略
        Args:
            vehicle_id: 车辆对ID
            failed_customer_id: 失败的客户ID
            constraint_analysis: 约束分析结果
        Returns:
            bool: 重规划是否成功
        """
        print(f"     执行无人机服务失败重规划（车辆对{vehicle_id}，失败客户{failed_customer_id}）")

        try:
            # Step 1: 找到失败客户所在的无人机任务
            failed_trip_info = self._locate_failed_drone_trip(vehicle_id, failed_customer_id)
            if not failed_trip_info:
                print(f"       未找到失败客户{failed_customer_id}所在的无人机任务")
                return False
            trip_idx = failed_trip_info['trip_index']
            trip = failed_trip_info['trip']
            failed_position = failed_trip_info['position_in_path']
            print(f"      📍 定位到任务{trip_idx}，客户在路径位置{failed_position}")
            # Step 2: 执行无人机重规划策略
            replanning_success = self._execute_drone_energy_replanning(
                vehicle_id, trip_idx, failed_customer_id, failed_position, constraint_analysis)
            if replanning_success:
                # Step 3: 重新计算时间和状态
                self._recalculate_drone_trip_states(vehicle_id, trip_idx)
                # Step 4: 验证重规划结果
                validation_success = self._validate_drone_replanning_result(vehicle_id, trip_idx)
                if validation_success:
                    print(f"      无人机重规划成功完成")
                    return True
                else:
                    print(f"       重规划验证失败，启动应急策略")
                    return self._drone_emergency_replanning(vehicle_id, trip_idx, failed_customer_id)
            else:
                print(f"       无人机重规划失败")
                return False
        except Exception as e:
            print(f"       无人机重规划异常: {e}")
            return False

    def _locate_failed_drone_trip(self, vehicle_id: int, failed_customer_id: int) -> Dict:
        """
        定位失败客户所在的无人机任务
        Returns:
            Dict: {trip_index, trip, position_in_path} 或 None
        """
        try:
            for trip_idx, trip in enumerate(self.DRONE_Routes[vehicle_id].route):
                if failed_customer_id in trip['path']:
                    position_in_path = trip['path'].index(failed_customer_id)
                    return {
                        'trip_index': trip_idx,
                        'trip': trip,
                        'position_in_path': position_in_path
                    }
            return None
        except Exception as e:
            print(f"        定位失败客户任务出错: {e}")
            return None

    def _execute_drone_energy_replanning(self, vehicle_id: int, trip_idx: int,
                                         failed_customer_id: int, failed_position: int,
                                         constraint_analysis: Dict) -> bool:
        """
        执行无人机能耗重规划核心逻辑
        策略：
        1. 检查按原路径继续是否可行
        2. 如果不可行，从末尾开始放弃客户
        3. 确保能安全返回
        """
        print(f"        🔋 开始能耗重规划分析...")

        try:
            trip = self.DRONE_Routes[vehicle_id].route[trip_idx]
            path = trip['path']
            current_battery = trip['current_remain_battery']

            print(f"          当前电池: {current_battery:.2f}/{self.drone_max_battery:.2f}")
            print(f"          原路径: {path}")

            # Step 1: 检查按原路径继续执行的能耗
            remaining_energy_needed = self._calculate_remaining_energy_after_failure(
                trip, failed_position, failed_customer_id)

            print(f"          继续执行需要能耗: {remaining_energy_needed:.2f}")

            # Step 2: 判断是否需要放弃客户
            if remaining_energy_needed <= current_battery:
                print(f"           能耗充足，继续执行原路径")
                # 只需要跳过失败客户，其他客户继续服务
                return self._skip_failed_customer_continue_path(vehicle_id, trip_idx, failed_customer_id)
            else:
                print(f"           能耗不足，开始逐步放弃客户")
                energy_deficit = remaining_energy_needed - current_battery
                print(f"          能耗缺口: {energy_deficit:.2f}")
                # Step 3: 从末尾开始放弃客户
                return self._abandon_customers_from_end(vehicle_id, trip_idx, failed_customer_id,
                                                        failed_position, energy_deficit)
        except Exception as e:
            print(f"           能耗重规划执行出错: {e}")
            return False

    def _skip_failed_customer_continue_path(self, vehicle_id: int, trip_idx: int, failed_customer_id: int) -> bool:
        """
        跳过失败客户，继续执行原路径
        """
        print(f"           标记客户{failed_customer_id}为跳过，继续后续服务")

        try:
            # 失败客户已经在之前被标记为 success=False
            # 这里只需要确保路径完整性和时间计算正确

            trip = self.DRONE_Routes[vehicle_id].route[trip_idx]
            # 重新计算路径中所有客户的服务时间（跳过失败客户）
            self._recalculate_drone_path_timing_skip_failed(trip, failed_customer_id)
            # 重新计算能耗（考虑失败客户的跳过）
            self._recalculate_trip_energy_with_skip(trip, failed_customer_id)
            print(f"          路径调整完成，跳过客户{failed_customer_id}")
            return True
        except Exception as e:
            print(f"          ❌ 跳过客户处理出错: {e}")
            return False

    def _abandon_customers_from_end(self, vehicle_id: int, trip_idx: int, failed_customer_id: int,
                                    failed_position: int, energy_deficit: float) -> bool:
        """从路径末尾开始逐个放弃客户直到能耗可行"""
        print(f"开始从末尾放弃客户（需减少能耗: {energy_deficit:.2f}）")

        try:
            trip = self.DRONE_Routes[vehicle_id].route[trip_idx]
            path = trip['path'].copy()

            # 获取可以放弃的客户
            candidates_to_abandon = []
            for i in range(failed_position + 1, len(path) - 1):  # 排除回收节点
                customer_id = path[i]
                if customer_id <= len(self.customers):
                    customer = self.customers[customer_id - 1]
                    if customer.success is None:  # 只能放弃未服务的客户
                        # 修正：正确定义priority
                        priority = 1 if customer.demand < 0 else 0  # 取货客户优先放弃
                        candidates_to_abandon.append({
                            'customer_id': customer_id,
                            'position': i,
                            'demand': customer.demand,
                            'priority': priority
                        })

            # 按优先级排序：取货客户优先放弃
            candidates_to_abandon.sort(key=lambda x: (x['priority'], -x['position']))

            print(f"可放弃客户候选: {[c['customer_id'] for c in candidates_to_abandon]}")

            if not candidates_to_abandon:
                print("没有可放弃的客户，尝试应急策略")
                return self._drone_emergency_replanning(vehicle_id, trip_idx, failed_customer_id)

            # 按从末尾到开头的顺序尝试放弃客户
            abandoned_customers = []
            current_path = path.copy()

            for candidate in reversed(candidates_to_abandon):  # 从末尾开始
                customer_id = candidate['customer_id']

                # 创建测试路径（移除该客户）
                test_path = [node for node in current_path if node != customer_id]

                # 重新计算测试路径的能耗
                test_energy = self._calculate_path_energy_after_failure(
                    test_path, failed_position, failed_customer_id, trip)

                current_battery = trip['current_remain_battery']
                print(f"测试放弃客户{customer_id}: 需要能耗{test_energy:.2f}")

                if test_energy <= current_battery:
                    # 能耗可行，确定放弃这些客户
                    abandoned_customers.append(customer_id)
                    current_path = test_path
                    print(f"放弃客户{customer_id}，能耗变为可行")
                    break
                else:
                    # 还需要继续放弃更多客户
                    abandoned_customers.append(customer_id)
                    current_path = test_path
                    print(f"暂定放弃客户{customer_id}，继续检查")

            # 检查最终路径是否可行
            final_energy = self._calculate_path_energy_after_failure(
                current_path, failed_position, failed_customer_id, trip)

            if final_energy <= trip['current_remain_battery']:
                # 执行客户放弃
                self._execute_customer_abandonment(vehicle_id, trip_idx, abandoned_customers, current_path)
                print(f"成功放弃{len(abandoned_customers)}个客户: {abandoned_customers}")
                print(f"最终能耗: {final_energy:.2f}/{trip['current_remain_battery']:.2f}")
                return True
            else:
                print("即使放弃所有可能客户仍无法满足能耗要求")
                return self._drone_emergency_replanning(vehicle_id, trip_idx, failed_customer_id)

        except Exception as e:
            print(f"放弃客户过程出错: {e}")
            return False

    def _execute_customer_abandonment(self, vehicle_id: int, trip_idx: int,
                                      abandoned_customers: List[int], new_path: List[int]):
        """执行客户放弃操作"""
        try:
            trip = self.DRONE_Routes[vehicle_id].route[trip_idx]

            # 更新路径
            trip['path'] = new_path

            # 重新计算载重（只保留未放弃的配送客户的载重）
            self._recalculate_trip_load_after_abandonment(trip, abandoned_customers)

            # 重新计算能耗
            self._recalculate_trip_energy_after_abandonment(trip)

            # 重置被放弃客户的服务状态
            for customer_id in abandoned_customers:
                if customer_id <= len(self.customers):
                    customer = self.customers[customer_id - 1]
                    customer.success = None  # 重置为未服务状态
                    customer.service_by = None  # 清除服务分配

            print(f"已执行客户放弃: {abandoned_customers}")

        except Exception as e:
            print(f"执行客户放弃操作出错: {e}")

    def _calculate_path_energy_after_failure(self, path: List[int], failed_position: int,
                                             failed_customer_id: int, trip: Dict) -> float:
        """
        计算失败后修改路径的总能耗
        """
        try:
            if len(path) <= 2:  # 只剩起飞和回收节点
                return 0.0
            # 创建临时trip用于能耗计算
            temp_trip = trip.copy()
            temp_trip['path'] = path
            # 从失败客户位置开始计算剩余能耗
            failed_pos_in_new_path = path.index(failed_customer_id) if failed_customer_id in path else failed_position
            remaining_energy = self._calculate_remaining_energy_after_failure(
                temp_trip, failed_pos_in_new_path, failed_customer_id)

            return remaining_energy

        except Exception as e:
            print(f"            计算路径能耗出错: {e}")
            return float('inf')  # 返回无穷大表示不可行

    def _recalculate_trip_load_after_abandonment(self, trip: Dict, abandoned_customers: List[int]):
        """
        重新计算放弃客户后的trip载重
        关键约束：无人机起飞后送货载重无法更改（包裹已装载在机上）
        """
        try:
            # 送货载重保持起飞时的初始值（包裹已在无人机上，无法卸载）
            delivery_load = trip['initial_load_delivery']  # 固定不变

            # 取货载重只计算未被放弃的取货客户
            pickup_load = 0

            for customer_id in trip['path'][1:-1]:  # 排除起终点
                if customer_id <= len(self.customers):
                    customer = self.customers[customer_id - 1]

                    # 只有取货客户且未被放弃才计入载重
                    if customer.demand < 0 and customer_id not in abandoned_customers:
                        pickup_load += abs(customer.demand)

            # 更新载重信息
            new_total_load = delivery_load + pickup_load

            trip['current_load'] = new_total_load
            trip['current_load_delivery'] = delivery_load  # 保持不变
            trip['current_load_pickup'] = pickup_load  # 重新计算
            # 注意：initial_load_delivery保持不变，因为包裹已经装载

            print(f"              载重更新: 送货{delivery_load}(固定), 取货{pickup_load}, 总计{new_total_load}")
            print(f"              被放弃客户: {abandoned_customers}")

        except Exception as e:
            print(f"              载重重新计算出错: {e}")

    def _recalculate_trip_load_after_abandonment(self, trip: Dict, abandoned_customers: List[int]):
        """
        重新计算放弃客户后的trip载重
        """
        try:
            # 从初始载重开始重新计算
            new_load_delivery = 0
            new_load_pickup = 0

            for customer_id in trip['path'][1:-1]:  # 排除起终点
                if customer_id <= len(self.customers):
                    customer = self.customers[customer_id - 1]
                    if customer.demand > 0:
                        new_load_delivery += customer.demand
                    else:
                        new_load_pickup += abs(customer.demand)

            # 更新载重信息
            trip['current_load'] = new_load_delivery
            trip['current_load_delivery'] = new_load_delivery
            trip['current_load_pickup'] = new_load_pickup
            trip['initial_load'] = new_load_delivery
            trip['initial_load_delivery'] = new_load_delivery

            print(f"              载重更新: 送货{new_load_delivery}, 取货{new_load_pickup}")

        except Exception as e:
            print(f"              载重重新计算出错: {e}")

    def _recalculate_trip_energy_after_abandonment(self, trip: Dict):
        """
        重新计算放弃客户后的trip能耗
        """
        try:
            if len(trip['path']) <= 2:
                trip['energy'] = 0
                return

            # 使用现有的能耗计算函数
            first_node_idx = trip['path'][0] - 1
            if first_node_idx >= 0:
                trip['energy'] = self.calculate_energy(
                    self.Vist_T[first_node_idx][4],
                    trip['path'],
                    trip['initial_load']
                )
            print(f"              能耗更新: {trip['energy']:.2f}")
        except Exception as e:
            print(f"              能耗重新计算出错: {e}")

    def _drone_emergency_replanning(self, vehicle_id: int, trip_idx: int, failed_customer_id: int) -> bool:
        """
        无人机应急重规划策略
        极端情况：只保留起飞和回收节点，放弃所有中间客户
        """
        print(f"          🆘 启动无人机应急重规划...")

        try:
            trip = self.DRONE_Routes[vehicle_id].route[trip_idx]
            original_path = trip['path'].copy()

            if len(original_path) <= 2:
                print(f"            路径已经是最简形式")
                return True

            # 保留起飞和回收节点
            launch_node = original_path[0]
            retrieval_node = original_path[-1]
            emergency_path = [launch_node, retrieval_node]

            # 将所有中间客户重置为未服务状态
            abandoned_customers = original_path[1:-1]

            print(f"            应急放弃所有中间客户: {abandoned_customers}")

            # 更新trip
            trip['path'] = emergency_path
            trip['current_load'] = 0
            trip['current_load_delivery'] = 0
            trip['current_load_pickup'] = 0
            trip['initial_load'] = 0
            trip['energy'] = 0  # 直接飞行无服务，能耗最小

            # 重置客户状态
            for customer_id in abandoned_customers:
                if customer_id <= len(self.customers):
                    customer = self.customers[customer_id - 1]
                    customer.success = None
                    customer.service_by = None

                    # 这些客户需要重新分配给其他车辆或路径
                    # 但在当前阶段，我们只是将其标记为未服务

            print(f"            ✅ 应急重规划完成，路径: {emergency_path}")

            # 将被放弃的客户添加到需要重新分配的列表中
            # (这里可以根据需要实现后续的重新分配逻辑)

            return True

        except Exception as e:
            print(f"            ❌ 应急重规划失败: {e}")
            return False

    def _recalculate_drone_trip_states(self, vehicle_id: int, trip_idx: int):
        """
        重新计算无人机任务的所有状态
        """
        try:
            trip = self.DRONE_Routes[vehicle_id].route[trip_idx]

            # 重新计算时间
            if len(trip['path']) > 2:
                launch_node_idx = trip['path'][0] - 1
                if launch_node_idx >= 0:
                    launch_time = self.Vist_T[launch_node_idx][4] if hasattr(self, 'Vist_T') else 0

                    # 重新计算路径中每个客户的到达和离开时间
                    self._update_drone_path_timing(trip, launch_time)

            # 更新卡车路径时间（如果需要）
            if hasattr(self, 'TRUCK_Routes') and vehicle_id < len(self.TRUCK_Routes):
                if len(self.TRUCK_Routes[vehicle_id].Troute) > 2:
                    self.Update_visit_T(vehicle_id, 1)

            print(f"          ✅ 任务状态重新计算完成")

        except Exception as e:
            print(f"          ❌ 状态重新计算出错: {e}")

    def _update_drone_path_timing(self, trip: Dict, launch_time: float):
        """
        更新无人机路径中所有客户的时间安排
        """
        try:
            path = trip['path']
            current_time = launch_time
            current_load = trip['initial_load']

            for i in range(1, len(path)):
                from_node = path[i - 1] - 1
                to_node = path[i] - 1

                # 计算飞行时间
                if from_node >= 0 and to_node >= 0:
                    distance = self.ALLdistanceDmatrix[from_node + 1][to_node + 1]
                    flight_time = distance / self.drone_speed
                    arrival_time = current_time + flight_time

                    # 更新客户时间信息
                    if to_node >= 0 and to_node < len(self.customers):
                        customer = self.customers[to_node]
                        customer.arrive_drone = arrival_time

                        if i < len(path) - 1:  # 不是回收节点
                            service_start = max(arrival_time, customer.start_time)
                            departure_time = service_start + self.service_time
                            customer.departure_drone = departure_time
                            current_time = departure_time

                            # 更新载重
                            if customer.success is not False:  # 未失败的客户
                                current_load -= customer.demand
                        else:  # 回收节点
                            customer.departure_drone = arrival_time
                            current_time = arrival_time

        except Exception as e:
            print(f"            更新路径时间出错: {e}")

    def _validate_drone_replanning_result(self, vehicle_id: int, trip_idx: int) -> bool:
        """
        验证无人机重规划结果的可行性
        """
        try:
            trip = self.DRONE_Routes[vehicle_id].route[trip_idx]

            # 检查能耗约束
            if trip['energy'] > self.drone_max_battery:
                print(f"          ❌ 验证失败：能耗超限 {trip['energy']:.2f} > {self.drone_max_battery}")
                return False

            # 检查载重约束
            if trip['current_load'] > self.drone_max_capacity:
                print(f"          ❌ 验证失败：载重超限 {trip['current_load']:.2f} > {self.drone_max_capacity}")
                return False

            # 检查路径完整性
            if len(trip['path']) < 2:
                print(f"          ❌ 验证失败：路径不完整 {trip['path']}")
                return False

            print(f"          ✅ 重规划结果验证通过")
            return True

        except Exception as e:
            print(f"          ❌ 验证过程出错: {e}")
            return False

    # ==================== 辅助函数和调用接口 ====================

    def _recalculate_drone_path_timing_skip_failed(self, trip: Dict, failed_customer_id: int):
        """
        重新计算无人机路径时间，跳过失败客户的服务
        """
        try:
            path = trip['path']
            if len(path) <= 2:
                return

            launch_node_idx = path[0] - 1
            if launch_node_idx < 0 or launch_node_idx >= len(self.customers):
                return

            # 获取起飞时间
            launch_time = self.Vist_T[launch_node_idx][4] if hasattr(self, 'Vist_T') else 0
            current_time = launch_time
            current_load = trip['initial_load']

            for i in range(1, len(path)):
                customer_id = path[i]
                prev_customer_id = path[i - 1]

                # 计算飞行时间
                distance = self.ALLdistanceDmatrix[prev_customer_id][customer_id]
                flight_time = distance / self.drone_speed
                arrival_time = current_time + flight_time

                if customer_id <= len(self.customers):
                    customer = self.customers[customer_id - 1]
                    customer.arrive_drone = arrival_time

                    if customer_id == failed_customer_id:
                        # 失败客户：到达但不服务，直接离开
                        customer.departure_drone = arrival_time
                        current_time = arrival_time
                    elif i < len(path) - 1:  # 不是回收节点
                        # 正常服务
                        service_start = max(arrival_time, customer.start_time)
                        departure_time = service_start + self.service_time
                        customer.departure_drone = departure_time
                        current_time = departure_time

                        # 更新载重（只有非失败客户才改变载重）
                        if customer.success is not False:
                            current_load -= customer.demand
                    else:
                        # 回收节点
                        customer.departure_drone = arrival_time

        except Exception as e:
            print(f"            重新计算路径时间（跳过失败客户）出错: {e}")

    def _recalculate_trip_energy_with_skip(self, trip: Dict, failed_customer_id: int):
        """
        重新计算考虑跳过失败客户的trip能耗
        """
        try:
            if len(trip['path']) <= 2:
                trip['energy'] = 0
                return

            path = trip['path']
            total_energy = 0
            current_load = trip['initial_load']

            for i in range(1, len(path)):
                from_node = path[i - 1]
                to_node = path[i]

                # 飞行能耗
                distance = self.ALLdistanceDmatrix[from_node][to_node]
                flight_time = distance / self.drone_speed
                flight_energy = flight_time * self.energy_fight * (current_load + self.drone_weight)
                total_energy += flight_energy

                # 如果不是最后一个节点，计算服务/等待能耗
                if i < len(path) - 1:
                    customer = self.customers[to_node - 1] if to_node <= len(self.customers) else None

                    if customer and to_node != failed_customer_id:
                        # 非失败客户：正常服务
                        if hasattr(customer, 'arrive_drone') and hasattr(customer, 'start_time'):
                            wait_time = max(0, customer.start_time - customer.arrive_drone)
                            total_energy += wait_time * self.energy_hover * (current_load + self.drone_weight)

                        # 服务能耗
                        total_energy += self.service_time * self.energy_service * (current_load + self.drone_weight)

                        # 更新载重
                        if customer.success is not False:
                            current_load -= customer.demand
                    # 失败客户：只有悬停等待的最小能耗，无服务能耗

            trip['energy'] = total_energy

        except Exception as e:
            print(f"            重新计算能耗（跳过失败客户）出错: {e}")

    # ==================== 主调用接口 ====================

    def integrate_drone_replanning_calls(self):
        """
        将无人机重规划集成到现有的动态优化流程中

        这个函数展示了如何在现有代码中调用无人机重规划功能
        """
        # 在你的 _comprehensive_constraint_analysis 函数中，当检测到无人机约束违反时：
        """
        # 示例调用方式（插入到现有的 run_dynamic_optimization 函数中）：

        if constraint_analysis['requires_replanning']:
            service_type = customer.service_by[0] if customer.service_by else None

            if service_type == "de":  # 无人机服务失败
                replan_success = self._drone_failure_specialized_replanning(
                    vehicle_id, current_customer_id, constraint_analysis)

                if replan_success:
                    print(f"     🚁 无人机重规划成功")
                    # 重新计算时间矩阵
                    self._recalculate_time_matrix_T()
                else:
                    print(f"     ❌ 无人机重规划失败，可能需要更激进的策略")
        """

    def _analyze_cascading_effects_within_vehicle(self, vehicle_id: int, failed_customer_id: int,
                                                  direct_violations: Dict) -> Dict:
        """
        分析车辆对内的连锁效应

        专门针对无人机服务失败的连锁效应分析
        """
        cascading_effects = {
            'drone_return_delay': [],
            'truck_schedule_impact': [],
            'subsequent_customer_impact': [],
            'energy_chain_effect': []
        }

        try:
            # 1. 检查无人机返回时间对卡车的影响
            drone_return_effects = self._analyze_drone_return_delay_effects(vehicle_id, failed_customer_id)
            if drone_return_effects:
                cascading_effects['drone_return_delay'] = drone_return_effects

            # 2. 检查对后续客户时间安排的影响
            subsequent_effects = self._analyze_subsequent_customer_timing_impact(vehicle_id, failed_customer_id)
            if subsequent_effects:
                cascading_effects['subsequent_customer_impact'] = subsequent_effects

            # 3. 检查能耗链式效应
            energy_effects = self._analyze_energy_chain_effects(vehicle_id, failed_customer_id)
            if energy_effects:
                cascading_effects['energy_chain_effect'] = energy_effects

            return cascading_effects

        except Exception as e:
            print(f"        连锁效应分析出错: {e}")
            return cascading_effects

    def _analyze_drone_return_delay_effects(self, vehicle_id: int, failed_customer_id: int) -> List[Dict]:
        """
        分析无人机返回延迟对卡车路径的影响
        """
        effects = []

        try:
            # 找到失败客户所在的无人机任务
            for trip_idx, trip in enumerate(self.DRONE_Routes[vehicle_id].route):
                if failed_customer_id in trip['path']:
                    retrieval_node = trip['retrieval_node']

                    # 检查回收节点在卡车路径中的位置
                    truck_route = self.TRUCK_Routes[vehicle_id].Troute
                    if retrieval_node in truck_route:
                        retrieval_position = truck_route.index(retrieval_node)

                        # 分析对后续卡车客户的时间影响
                        for i in range(retrieval_position + 1, len(truck_route) - 1):
                            subsequent_customer = truck_route[i]
                            effects.append({
                                'type': 'truck_schedule_delay',
                                'affected_customer': subsequent_customer,
                                'cause': f'无人机任务{trip_idx}返回延迟'
                            })

                    break

        except Exception as e:
            print(f"          分析无人机返回延迟效应出错: {e}")

        return effects

    def _has_critical_violations(self, direct_violations: Dict) -> bool:
        """
        判断是否存在关键约束违反
        """
        # 能耗不足是关键违反
        if 'drone_energy' in direct_violations:
            energy_violations = direct_violations['drone_energy']
            if energy_violations.get('energy_insufficient') or energy_violations.get('severity') == 'critical':
                return True

        # 严重载重违反是关键违反
        if 'drone_load' in direct_violations:
            load_violations = direct_violations['drone_load']
            if load_violations.get('has_violations') and load_violations.get('max_violation',
                                                                             0) > self.drone_max_capacity * 0.1:
                return True

        if 'truck_load' in direct_violations:
            truck_violations = direct_violations['truck_load']
            if truck_violations.get('truck_load_violations'):
                return True

        return False

    def _has_critical_cascading_effects(self, cascading_effects: Dict) -> bool:
        """
        判断是否存在关键连锁效应
        """
        # 如果有大量客户受到连锁影响，需要重规划
        total_affected = 0
        for effect_type, effects in cascading_effects.items():
            total_affected += len(effects) if isinstance(effects, list) else 0

        # 如果超过3个后续客户受影响，认为是关键连锁效应
        return total_affected > 3

# ==================== 无人机重规划结束   ===================

    def _truck_failure_specialized_replanning(self, vehicle_id: int, failed_customer_id: int,
                                              constraint_analysis: Dict) -> bool:
        """
        阶段3：卡车服务失败专门重规划策略
        基于ALNS算法框架，严格限制在指定车辆对内操作

        Args:
            vehicle_id: 车辆对ID
            failed_customer_id: 失败的客户ID
            constraint_analysis: 约束分析结果

        Returns:
            bool: 重规划是否成功
        """
        print(f"     🚚 执行卡车服务失败重规划（车辆对{vehicle_id}，失败客户{failed_customer_id}）")

        try:
            # Step 1: 确定重规划范围（从失败客户开始的所有后续客户）
            customers_to_replan = self._determine_truck_replanning_scope(vehicle_id, failed_customer_id)

            if not customers_to_replan:
                print(f"       没有需要重规划的客户")
                return True

            print(f"       重规划范围: {len(customers_to_replan)}个客户 - {customers_to_replan}")

            # Step 2: 记录重规划前状态
            initial_cost = self.cost_single_vehicle(vehicle_id)
            print(f"       重规划前成本: {initial_cost:.2f}")

            # Step 3: 执行ALNS重规划主循环
            best_solution_found = self._execute_truck_alns_replanning(
                vehicle_id, customers_to_replan, constraint_analysis)

            if best_solution_found:
                # Step 4: 最终可行性检查和修复
                final_feasibility = self._final_feasibility_repair(vehicle_id)

                # Step 5: 计算最终成本和改进
                final_cost = self.cost_single_vehicle(vehicle_id)
                improvement = initial_cost - final_cost

                print(f"       重规划后成本: {final_cost:.2f}")
                print(f"       成本改进: {improvement:.2f}")

                if final_feasibility:
                    print(f"       ✅ 卡车重规划成功完成")
                    return True
                else:
                    print(f"       ⚠️ 可行性检查失败，但保留当前解")
                    return True  # 接受次优解而不是完全失败
            else:
                print(f"       ❌ ALNS重规划未找到改进解，执行应急策略")
                return self._truck_emergency_replanning(vehicle_id, customers_to_replan)

        except Exception as e:
            print(f"       ❌ 卡车重规划异常: {e}")
            return self._truck_emergency_replanning(vehicle_id, customers_to_replan)

    def _determine_truck_replanning_scope(self, vehicle_id: int, failed_customer_id: int) -> List[int]:
        """
        确定卡车重规划的范围
        包括：1) 失败客户之后的卡车客户  2) 相关的无人机任务客户

        Args:
            vehicle_id: 车辆对ID
            failed_customer_id: 失败客户ID

        Returns:
            需要重新规划的客户列表
        """
        customers_to_replan = []

        try:
            truck_route = self.TRUCK_Routes[vehicle_id].Troute

            if failed_customer_id not in truck_route:
                print(f"         失败客户{failed_customer_id}不在车辆对{vehicle_id}的卡车路径中")
                return customers_to_replan

            failed_position = truck_route.index(failed_customer_id)
            print(f"         失败客户在卡车路径位置: {failed_position}")

            # 1. 收集失败客户之后的所有卡车客户（包括失败客户本身）
            truck_customers_after_failure = truck_route[failed_position:-1]  # 排除终点仓库
            customers_to_replan.extend(truck_customers_after_failure)

            print(f"         卡车后续客户: {truck_customers_after_failure}")

            # 2. 收集受影响的无人机任务客户
            affected_drone_customers = self._collect_affected_drone_customers(
                vehicle_id, failed_position, truck_route)
            customers_to_replan.extend(affected_drone_customers)

            print(f"         受影响无人机客户: {affected_drone_customers}")

            # 3. 去重并验证所有客户都属于当前车辆对
            customers_to_replan = list(set(customers_to_replan))
            vehicle_customers = self.get_vehicle_customers(vehicle_id)
            valid_customers = [c for c in customers_to_replan if c in vehicle_customers]

            if len(valid_customers) != len(customers_to_replan):
                invalid = set(customers_to_replan) - set(valid_customers)
                print(f"         过滤跨车辆对客户: {invalid}")

            return valid_customers

        except Exception as e:
            print(f"         确定重规划范围出错: {e}")
            return []

    def _collect_affected_drone_customers(self, vehicle_id: int, failed_position: int,
                                          truck_route: List[int]) -> List[int]:
        """
        收集受卡车失败影响的无人机客户
        主要是失败位置之后的无人机任务客户
        """
        affected_customers = []

        try:
            # 获取失败位置之后的卡车节点（潜在的无人机起飞/回收节点）
            subsequent_truck_nodes = truck_route[failed_position + 1:-1]

            # 检查每个无人机任务是否受到影响
            for trip in self.DRONE_Routes[vehicle_id].route:
                launch_node = trip['launch_node']
                retrieval_node = trip['retrieval_node']

                # 如果无人机的起飞或回收节点在失败位置之后，则该任务受影响
                if (launch_node in subsequent_truck_nodes or
                        retrieval_node in subsequent_truck_nodes):
                    # 将该任务的所有客户加入重规划范围
                    drone_customers = trip['path'][1:-1]  # 排除起终点
                    affected_customers.extend(drone_customers)
                    print(f"         无人机任务受影响: 起飞{launch_node}, 回收{retrieval_node}")
                    print(f"           任务客户: {drone_customers}")

            return affected_customers

        except Exception as e:
            print(f"         收集无人机影响客户出错: {e}")
            return []

    def _execute_truck_alns_replanning(self, vehicle_id: int, customers_to_replan: List[int],
                                       constraint_analysis: Dict) -> bool:
        """
        执行基于ALNS的卡车重规划主循环

        Args:
            vehicle_id: 车辆对ID
            customers_to_replan: 需要重规划的客户列表
            constraint_analysis: 约束分析结果

        Returns:
            是否找到改进解
        """
        print(f"       🔄 启动ALNS重规划主循环...")

        # ALNS参数设置
        max_iterations = min(50, len(customers_to_replan) * 5)  # 自适应迭代次数
        max_no_improve = max(10, max_iterations // 5)
        temperature_start = self.cost_single_vehicle(vehicle_id) * 0.1
        temperature_decay = 0.95

        # 初始化
        best_cost = self.cost_single_vehicle(vehicle_id)
        current_cost = best_cost
        temperature = temperature_start
        iterations_no_improve = 0

        print(f"         ALNS参数: max_iter={max_iterations}, temp_start={temperature_start:.2f}")

        try:
            for iteration in range(max_iterations):
                # Step 1: 选择并执行摧毁算子
                destroy_operator = self.destroy_ops.select_destroy_operator()
                customers_removed = getattr(self.destroy_ops, f"{destroy_operator}_removal")(
                    vehicle_id, customers_to_replan[0] if customers_to_replan else -1)

                if not customers_removed:
                    print(f"           迭代{iteration}: 摧毁算子{destroy_operator}未移除任何客户")
                    continue

                print(f"           迭代{iteration}: {destroy_operator}移除{len(customers_removed)}个客户")

                # Step 2: 选择并执行修复算子
                repair_success = self.repair_ops.repair_solution(vehicle_id, customers_removed)

                if not repair_success:
                    print(f"           修复失败，跳过此次迭代")
                    continue

                # Step 3: 轻量级局部搜索（条件触发）
                new_cost = self.cost_single_vehicle(vehicle_id)
                if new_cost < current_cost * 1.05:  # 如果解质量较好，进行局部搜索
                    optimized_cost = self.local_search(vehicle_id, new_cost)
                    if optimized_cost < new_cost:
                        new_cost = optimized_cost
                        print(f"           局部搜索改进: {new_cost - optimized_cost:.2f}")

                # Step 4: 接受准则（模拟退火）
                accept_solution = False
                if new_cost < best_cost:
                    # 新的最优解
                    accept_solution = True
                    best_cost = new_cost
                    iterations_no_improve = 0
                    print(f"           ✅ 新最优解: {new_cost:.2f} (改进{current_cost - new_cost:.2f})")

                    # 更新信息素
                    self.update_pheromone(current_cost, new_cost)

                elif new_cost < current_cost:
                    # 局部改进
                    accept_solution = True
                    iterations_no_improve = 0
                    print(f"           📈 局部改进: {new_cost:.2f}")

                else:
                    # 模拟退火接受
                    if temperature > 0:
                        accept_prob = math.exp((current_cost - new_cost) / temperature)
                        if random.random() < accept_prob:
                            accept_solution = True
                            print(f"           🌡️ 模拟退火接受: {accept_prob:.3f}")

                # Step 5: 更新当前解
                if accept_solution:
                    current_cost = new_cost
                else:
                    # 拒绝解，恢复之前状态（这里简化处理）
                    iterations_no_improve += 1

                # Step 6: 更新算子权重和信息素
                improved = new_cost < best_cost
                self.destroy_ops.update_operator_performance(destroy_operator, improved)

                # Step 7: 降温和终止条件检查
                temperature *= temperature_decay

                if iterations_no_improve >= max_no_improve:
                    print(f"           连续{max_no_improve}次无改进，提前终止")
                    break

            # 信息素挥发
            if hasattr(self, 'evaporate_pheromone'):
                self.evaporate_pheromone()

            improvement = self.cost_single_vehicle(vehicle_id) - best_cost
            print(f"         ALNS完成，最终改进: {improvement:.2f}")

            return improvement > 0.01  # 显著改进才认为成功

        except Exception as e:
            print(f"         ALNS执行异常: {e}")
            return False

    def _final_feasibility_repair(self, vehicle_id: int) -> bool:
        """
        最终的可行性检查和修复
        """
        print(f"         🔧 执行最终可行性修复...")

        try:
            if hasattr(self, 'feasibility_repair_ops') and self.feasibility_repair_ops:
                return self.feasibility_repair_ops.check_and_repair_feasibility(vehicle_id)
            else:
                # 简化的可行性检查
                return self._basic_feasibility_check(vehicle_id)

        except Exception as e:
            print(f"         可行性修复出错: {e}")
            return False

    def _basic_feasibility_check(self, vehicle_id: int) -> bool:
        """
        基础可行性检查
        """
        try:
            # 检查卡车载重
            truck = self.TRUCK_Routes[vehicle_id]
            if truck.current_load > truck.max_capacity:
                print(f"         ❌ 卡车载重超限: {truck.current_load} > {truck.max_capacity}")
                return False

            # 检查无人机约束
            for trip in self.DRONE_Routes[vehicle_id].route:
                if trip.get('current_load', 0) > self.drone_max_capacity:
                    print(f"         ❌ 无人机载重超限")
                    return False
                if trip.get('energy', 0) > self.drone_max_battery:
                    print(f"         ❌ 无人机能耗超限")
                    return False

            return True

        except Exception as e:
            print(f"         基础可行性检查出错: {e}")
            return False

    def _truck_emergency_replanning(self, vehicle_id: int, customers_to_replan: List[int]) -> bool:
        """
        卡车应急重规划策略
        将所有需要重规划的客户简单插入到卡车路径末尾
        """
        print(f"         🆘 启动卡车应急重规划...")

        try:
            if not customers_to_replan:
                return True

            # 1. 从所有路径中移除这些客户
            self._remove_customers_from_vehicle_routes(vehicle_id, customers_to_replan)

            # 2. 将所有客户插入到卡车路径末尾（返回仓库前）
            truck_route = self.TRUCK_Routes[vehicle_id].Troute
            insert_position = len(truck_route) - 1

            inserted_count = 0
            for customer_id in customers_to_replan:
                # 验证客户属于当前车辆对
                if self.validate_customer_assignment(vehicle_id, customer_id):
                    truck_route.insert(insert_position, customer_id)
                    insert_position += 1

                    # 更新客户服务信息
                    self.customers[customer_id - 1].service_by = ["tk", vehicle_id]
                    inserted_count += 1

                    print(f"           应急插入客户{customer_id}")

            print(f"         应急重规划完成: {inserted_count}/{len(customers_to_replan)}个客户")

            # 3. 重新计算时间和载重
            if len(truck_route) > 2:
                self.Update_visit_T(vehicle_id, 1)

            return inserted_count > 0

        except Exception as e:
            print(f"         ❌ 应急重规划失败: {e}")
            return False

    def _analyze_subsequent_customer_timing_impact(self, vehicle_id: int, failed_customer_id: int) -> List[Dict]:
        """分析对后续客户时间安排的影响"""
        effects = []
        try:
            # 检查同一车辆对中失败客户之后的客户
            truck_route = self.TRUCK_Routes[vehicle_id].Troute
            if failed_customer_id in truck_route:
                failed_pos = truck_route.index(failed_customer_id)
                for i in range(failed_pos + 1, len(truck_route) - 1):
                    effects.append({
                        'type': 'timing_delay',
                        'affected_customer': truck_route[i],
                        'vehicle_type': 'truck'
                    })

            # 检查无人机任务
            for trip in self.DRONE_Routes[vehicle_id].route:
                if failed_customer_id in trip['path']:
                    failed_pos = trip['path'].index(failed_customer_id)
                    for i in range(failed_pos + 1, len(trip['path']) - 1):
                        effects.append({
                            'type': 'timing_delay',
                            'affected_customer': trip['path'][i],
                            'vehicle_type': 'drone'
                        })

            return effects
        except Exception as e:
            print(f"          分析后续客户时间影响出错: {e}")
            return []

    def _analyze_energy_chain_effects(self, vehicle_id: int, failed_customer_id: int) -> List[Dict]:
        """分析能耗链式效应"""
        effects = []
        try:
            # 简化实现：如果能耗接近上限，标记为链式效应
            for trip_idx, trip in enumerate(self.DRONE_Routes[vehicle_id].route):
                if failed_customer_id in trip['path']:
                    current_energy = trip.get('energy', 0)
                    if current_energy > self.drone_max_battery * 0.9:
                        effects.append({
                            'type': 'energy_critical',
                            'trip_index': trip_idx,
                            'energy_ratio': current_energy / self.drone_max_battery
                        })
            return effects
        except Exception as e:
            print(f"          分析能耗链式效应出错: {e}")
            return []

    def _determine_violation_severity(self, direct_violations: Dict, cascading_effects: Dict) -> str:
        """确定约束违反的严重程度"""
        # 检查关键违反
        if self._has_critical_violations(direct_violations):
            return 'critical'

        # 检查中等违反
        violation_count = sum(len(v) if isinstance(v, list) else (1 if v else 0)
                              for v in direct_violations.values())
        cascading_count = sum(len(v) if isinstance(v, list) else (1 if v else 0)
                              for v in cascading_effects.values())

        if violation_count > 2 or cascading_count > 3:
            return 'moderate'
        elif violation_count > 0 or cascading_count > 0:
            return 'minor'
        else:
            return 'none'

# ==================== 卡车重规划   ===================



# ==================== 卡车重规划结束   ===================


    def _initialize_time_matrix_T(self):
        """
        初始化时间矩阵T：收集所有客户的服务时间并排序
        """
        print(" 初始化时间矩阵T...")

        # 首先确保所有客户都有正确的服务时间
        self._calculate_all_service_times()

        # 收集所有客户的 [客户ID, 服务开始时间]
        customer_times = []
        for customer in self.customers:
            service_time = getattr(customer, 'service_begin', customer.start_time)
            customer_times.append([customer.cust_no, service_time])
        # 按服务时间排序
        customer_times.sort(key=lambda x: x[1])
        # 构建T矩阵
        self.T = np.array(customer_times, dtype='object')
        print(f"    时间矩阵T构建完成: {len(customer_times)}个客户")
        if len(customer_times) > 0:
            print(f"    服务时间范围: {self.T[0][1]:.2f} - {self.T[-1][1]:.2f}")

    def _recalculate_time_matrix_T(self):
        """
        重规划后重新计算时间矩阵T
        """
        print("     重新计算时间矩阵T...")
        # 1. 重新计算所有客户的服务时间
        self._calculate_all_service_times()
        # 2. 重新排序构建T矩阵
        self._initialize_time_matrix_T()
        print(f"   时间矩阵T更新完成: {len(self.T)}个客户")

    def _calculate_all_service_times(self):
        """
        重新计算所有客户的服务时间
        """
        # 为每个车辆对重新计算时间
        for truck_id in range(len(self.TRUCK_Routes)):
            # 更新该车辆对的访问时间矩阵
            if len(self.TRUCK_Routes[truck_id].Troute) > 2:  # 有客户
                self.Update_visit_T(truck_id, 1)  # 从第一个客户开始更新

            # 重新计算无人机任务的时间和能耗
            for trip in self.DRONE_Routes[truck_id].route:
                if len(trip['path']) > 2:  # 有客户
                    # 重新计算能耗
                    first_node_idx = trip['path'][0] - 1
                    if first_node_idx >= 0:
                        trip['energy'] = self.calculate_energy(
                            self.Vist_T[first_node_idx][4],
                            trip['path'],
                            trip.get('current_load', 0)
                        )

        # 更新所有客户的service_begin时间
        for customer in self.customers:
            if hasattr(customer, 'service_by') and customer.service_by:
                if customer.service_by[0] == "tk":
                    # 卡车服务：使用arrive_truck时间
                    customer.service_begin = getattr(customer, 'arrive_truck', customer.start_time)
                else:
                    # 无人机服务：使用arrive_drone时间
                    customer.service_begin = getattr(customer, 'arrive_drone', customer.start_time)
            else:
                customer.service_begin = customer.start_time

    def _find_next_unprocessed_stage(self, processed_customers):
        """
        在更新后的T矩阵中找到下一个未处理的客户
        """
        for stage_idx in range(len(self.T)):
            customer_id = int(self.T[stage_idx][0])
            if customer_id not in processed_customers:
                return stage_idx
        return None

    def _emergency_replan_within_vehicle(self, vehicle_id, customers_to_replan):
        """
        车辆对内的应急重规划策略
        将所有客户简单插入到卡车路径的末尾
        """
        try:
            print(f"       🆘 执行车辆对{vehicle_id}应急重规划...")

            truck_route = self.TRUCK_Routes[vehicle_id].Troute
            insert_position = len(truck_route) - 1  # 在返回仓库前插入

            inserted_count = 0
            for customer_id in customers_to_replan:
                # 验证客户属于当前车辆对
                if self.validate_customer_assignment(vehicle_id, customer_id):
                    truck_route.insert(insert_position, customer_id)
                    insert_position += 1

                    # 更新客户服务信息
                    self.customers[customer_id - 1].service_by = ["tk", vehicle_id]
                    inserted_count += 1

                    print(f"         应急插入客户{customer_id}")

            print(f"       ✅ 应急重规划完成: {inserted_count}/{len(customers_to_replan)}个客户")
            return inserted_count > 0

        except Exception as e:
            print(f"       ❌ 应急重规划失败: {e}")
            return False

    def _remove_customers_from_vehicle_routes(self, vehicle_id: int, customers_to_remove: list):
        """
        从指定车辆对的路径中删除客户（严格限制在车辆对内）
        """
        vehicle_customers = self.get_vehicle_customers(vehicle_id)

        # 验证所有要删除的客户都属于当前车辆对
        invalid_customers = [c for c in customers_to_remove if c not in vehicle_customers]
        if invalid_customers:
            print(f"         ⚠️ 发现不属于车辆对{vehicle_id}的客户: {invalid_customers}")
            customers_to_remove = [c for c in customers_to_remove if c in vehicle_customers]

        # 从卡车路径中删除
        truck_route = self.TRUCK_Routes[vehicle_id].Troute
        for customer_id in customers_to_remove:
            if customer_id in truck_route:
                truck_route.remove(customer_id)
                print(f"           从车辆对{vehicle_id}卡车路径中删除客户{customer_id}")

        # 从无人机路径中删除
        drone_routes = self.DRONE_Routes[vehicle_id].route
        trips_to_remove = []

        for trip_idx, trip in enumerate(drone_routes):
            customers_in_trip = set(trip['path'][1:-1])  # 排除起终点
            customers_to_delete = customers_in_trip & set(customers_to_remove)

            if customers_to_delete:
                for customer_id in customers_to_delete:
                    if customer_id in trip['path']:
                        trip['path'].remove(customer_id)
                        print(f"           从车辆对{vehicle_id}无人机任务{trip_idx}中删除客户{customer_id}")

                # 如果路径只剩起终点，标记删除整个trip
                if len(trip['path']) <= 2:
                    trips_to_remove.append(trip_idx)
                    print(f"           标记删除车辆对{vehicle_id}的空无人机任务{trip_idx}")

        # 逆序删除空的trips
        for trip_idx in reversed(trips_to_remove):
            del drone_routes[trip_idx]
            print(f"           删除车辆对{vehicle_id}的空无人机任务{trip_idx}")

    def _emergency_replan_within_vehicle(self, vehicle_id: int, customers_to_replan: list) -> bool:
        """
        应急重规划策略
        """
        try:
            # 简单策略：将所有客户插入到该车辆对的卡车路径末尾（返回仓库前）
            truck_route = self.TRUCK_Routes[vehicle_id].Troute
            insert_position = len(truck_route) - 1  # 在返回仓库前插入
            for customer_id in customers_to_replan:
                truck_route.insert(insert_position, customer_id)
                insert_position += 1
                # 更新客户服务信息
                self.customers[customer_id - 1].service_by = ["tk", vehicle_id]
            print(f"    应急重规划完成，{len(customers_to_replan)}个客户已插入卡车路径")
            return True
        except Exception as e:
            print(f"   ❌ 应急重规划失败: {e}")
            return False

    def _initialize_customer_success_status(self):
        """初始化所有客户的成功状态为None（待服务）"""
        for customer in self.customers:
            customer.success = None  # None=待服务, True=成功, False=失败
        print(f"初始化{len(self.customers)}个客户的服务状态为待服务")

    # 计算卡车\无人机行驶路径距离
    def Distance(self):
        customers_array = np.array([[customer.xcoord, customer.ycoord] for customer in self.customers])
        customers_array = np.insert(customers_array, 0, [40,50], 0)                          # 将选定的 仓库 加到路径的起始位置
        customer_count = len(customers_array)                                                               # 客户数量
        self.Tdis = [[0] * customer_count for i in range(customer_count)]                                   # 初始化距离矩阵
        for i in range(customer_count):                                                                     # 对每一个城市
            for j in range(customer_count):                                                                 # 对每一个城市
                if i != j:                                                                                  # 如果不是同一个城市
                    self.Tdis[i][j] = abs(customers_array[i][0] - customers_array[j][0]) + abs(
                        customers_array[i][1] - customers_array[j][1])                                      # 计 算 距 离
                else:
                    self.Tdis[i][j] = 0                                                                     # 同一个城市距离为 0
        self.Ddis = [[0] * customer_count for i in range(customer_count)]
        for i in range(customer_count):
            for j in range(customer_count):
                if i != j:                                                                                  # 如果不是同一个城市
                    self.Ddis[i][j] = math.sqrt((customers_array[i][0] - customers_array[j][0]) ** 2 + (
                            customers_array[i][1] - customers_array[j][1]) ** 2)                            # 计算距离
                else:
                    self.Ddis[i][j] = 0

    # 根据初始解初始化卡车\无人机的载重
    def Initial_vehicle_information(self):
        """车辆信息初始化"""
        for idx, (truck, drone) in enumerate(zip(self.TRUCK_Routes, self.DRONE_Routes), start=1):
            # 正确计算卡车初始载重
            truck_delivery_load = 0
            truck_pickup_load = 0
            # 计算卡车直接服务的客户载重
            for customer_id in truck.Troute[1:-1]:  # 排除起终点
                customer = self.customers[customer_id - 1]
                if customer.demand > 0:
                    truck_delivery_load += customer.demand
                else:
                    truck_pickup_load += abs(customer.demand)
            # 计算无人机任务的载重（需要卡车携带）
            for trip in drone.route:
                for customer_id in trip['path'][1:-1]:
                    customer = self.customers[customer_id - 1]
                    if customer.demand > 0:
                        truck_delivery_load += customer.demand
            # 设置卡车载重信息
            truck.initial_load = truck_delivery_load
            truck.initial_load_delivery = truck_delivery_load
            truck.current_load = truck_delivery_load
            truck.current_load_delivery = truck_delivery_load
            truck.current_load_pickup = 0
            print(f" 卡车{idx}载重初始化: 配送载重={truck_delivery_load}")
            # 计算无人机载重信息
            for trip in drone.route:
                trip_delivery_load = 0
                for customer_id in trip['path'][1:-1]:
                    customer = self.customers[customer_id - 1]
                    if customer.demand > 0:
                        trip_delivery_load += customer.demand
                trip['current_load'] = trip_delivery_load
                trip['initial_load'] = trip_delivery_load
                trip['current_load_delivery'] = trip_delivery_load

    def Initial_visit_T(self):
        #dtype='object'：这里我们将 Vist_T 数组初始化为 object 类型，这样它就能接受任意类型的数据（整数和浮动数）。这可以确保在赋值过程中不会发生类型冲突。
        self.Vist_T = np.empty((self.cnum, 5), dtype='object')  # 用 object 类型初始化，避免重复赋值
        # 填充数据
        for i in range(self.cnum):
            self.Vist_T[i, 0] = self.customers[i].cust_no                        # 第一列为整数类型
            self.Vist_T[i, 1] = self.customers[i].arrive_truck                   # 卡车到达时间
            self.Vist_T[i, 2] = self.customers[i].departure_truck                # 卡车离开时间
            self.Vist_T[i, 3] = self.customers[i].arrive_drone                   # 无人机到达时间
            self.Vist_T[i, 4] = self.customers[i].departure_drone                # 无人机离开时间

    def set_customer_service_status(self, customer_id: int, success_status: bool):
        """
        统一设置客户服务状态
        Args:
            customer_id: 客户ID
            success_status: True=成功, False=失败
        """
        if customer_id <= len(self.customers):
            self.customers[customer_id - 1].success = success_status
            print(f"    客户{customer_id}服务状态设置为: {'成功' if success_status else '失败'}")

    def Update_visit_T(self, truck_id, customer_index):                          # 更新路径中的客户时间
        launch_node=[]
        retrieval_node=[]
        if self.DRONE_Routes[truck_id].route:
            launch_node = [trip['launch_node'] for trip in self.DRONE_Routes[truck_id].route]                        # 起飞节点集合
            retrieval_node = [trip['retrieval_node'] for trip in self.DRONE_Routes[truck_id].route]                  # 回收节点集合
        for j in range(customer_index, len(self.TRUCK_Routes[truck_id].Troute)-1):                                   # 从传入节点开始更新时间
            if j-1==0:
                j_index =  self.TRUCK_Routes[truck_id].Troute[j]-1
                distance=self.Tdis[0][j_index+1]
                self.Vist_T[j_index][1]=distance/self.truck_speed                                                                               #更新卡车到达时间
                self.Vist_T[j_index][2]=max(self.Vist_T[j_index][1], self.customers[self.Vist_T[j_index][0]-1].start_time)+self.service_time    #更新卡车离开时间
                if self.TRUCK_Routes[truck_id].Troute[j] in launch_node:                                                                        #假设 当前节点仅为起飞节点时
                    self.Vist_T[j_index][3] = self.Vist_T[j_index][1]
                    self.Vist_T[j_index][4] = self.Vist_T[j_index][1]
                    for trip in self.DRONE_Routes[truck_id].route:                                                                              #更新以当前节点为起飞节点的无人机路径
                        if trip['launch_node'] == self.TRUCK_Routes[truck_id].Troute[j]:
                            path=trip['path']
                            for i in range(1, len(path)):
                                prev_indices =  path[i - 1]-1
                                current_indices =  path[i]-1
                                distance = self.Ddis[prev_indices+1][current_indices+1]
                                self.Vist_T[current_indices][3] = self.Vist_T[prev_indices][4]+distance/self.drone_speed
                                self.Vist_T[current_indices][4] = max(self.Vist_T[current_indices][3], self.customers[self.Vist_T[current_indices][0]-1].start_time)+self.service_time
                                if path[i] not in retrieval_node:
                                    self.Vist_T[current_indices][1] = 0
                                    self.Vist_T[current_indices][2] = 0
                else:
                    self.Vist_T[j_index][3] = self.Vist_T[j_index][1]
                    self.Vist_T[j_index][4] = self.Vist_T[j_index][2]
            else:
                j_index = self.TRUCK_Routes[truck_id].Troute[j]-1
                prev_indices =  self.TRUCK_Routes[truck_id].Troute[j-1]-1
                distance = self.Tdis[prev_indices+1][j_index+1]
                self.Vist_T[j_index][1] = distance / self.truck_speed+self.Vist_T[prev_indices][2]
                self.Vist_T[j_index][2] = max(self.Vist_T[j_index][1], self.customers[self.Vist_T[j_index][0] - 1].start_time) + self.service_time
                # 判断当前节点 卡车不搭载无人机
                if (self.TRUCK_Routes[truck_id].Troute[j - 1] in launch_node and self.TRUCK_Routes[truck_id].Troute[j] not in retrieval_node) or (self.Vist_T[prev_indices][3] == 0 and self.TRUCK_Routes[truck_id].Troute[j] not in retrieval_node):
                    self.Vist_T[j_index][3] = 0
                    self.Vist_T[j_index][4] = 0
                else:
                    if self.TRUCK_Routes[truck_id].Troute[j] not in launch_node and self.TRUCK_Routes[truck_id].Troute[j] not in retrieval_node:       #当前节点为普通客户节点时
                        self.Vist_T[j_index][3] = self.Vist_T[j_index][1]
                        self.Vist_T[j_index][4] = self.Vist_T[j_index][2]
                    if self.TRUCK_Routes[truck_id].Troute[j] in launch_node and self.TRUCK_Routes[truck_id].Troute[j] not in retrieval_node:           #当前节点仅为起飞节点时
                        self.Vist_T[j_index][3] = self.Vist_T[j_index][1]
                        self.Vist_T[j_index][4] = self.Vist_T[j_index][1]
                        for trip in self.DRONE_Routes[truck_id].route:
                            if trip['launch_node'] == self.TRUCK_Routes[truck_id].Troute[j]:
                                path = trip['path']
                                for i in range(1, len(path)):
                                    prev_indices =  path[i - 1]-1
                                    current_indices =  path[i]-1
                                    distance = self.Ddis[prev_indices+1][current_indices+1]
                                    self.Vist_T[current_indices][3] = self.Vist_T[prev_indices][4] + distance / self.drone_speed
                                    self.Vist_T[current_indices][4] = max(self.Vist_T[current_indices][3],
                                                       self.customers[self.Vist_T[current_indices][0] - 1].start_time) + self.service_time
                                    if path[i] not in retrieval_node:
                                        self.Vist_T[current_indices][1] = 0
                                        self.Vist_T[current_indices][2] = 0
                    if self.TRUCK_Routes[truck_id].Troute[j] in launch_node and self.TRUCK_Routes[truck_id].Troute[j] in retrieval_node:               #当前节点既为起飞节点又为回收节点时
                        max_time=max(self.Vist_T[j_index][2], self.Vist_T[j_index][3])
                        self.Vist_T[j_index][2] = max_time
                        self.Vist_T[j_index][4] = self.Vist_T[j_index][3]
                        for trip in self.DRONE_Routes[truck_id].route:
                            if trip['launch_node'] == self.TRUCK_Routes[truck_id].Troute[j]:
                                path = trip['path']
                                for i in range(1, len(path)):
                                    prev_indices =  path[i - 1]-1
                                    current_indices =  path[i]-1
                                    distance = self.Ddis[prev_indices+1][current_indices+1]
                                    self.Vist_T[current_indices][3] = self.Vist_T[i - 1][4] + distance / self.drone_speed
                                    self.Vist_T[current_indices][4] = max(self.Vist_T[current_indices][3], self.customers[self.Vist_T[current_indices][0] - 1].start_time) + self.service_time
                                    if path[i] not in retrieval_node:
                                        self.Vist_T[current_indices][1] = 0
                                        self.Vist_T[current_indices][2] = 0
                    if self.TRUCK_Routes[truck_id].Troute[j] not in launch_node and self.TRUCK_Routes[truck_id].Troute[j] in retrieval_node:  # 当前节点仅为回收节点时
                        max_time = max(self.Vist_T[j_index][2], self.Vist_T[j_index][3])
                        self.Vist_T[j_index][2] = max_time
                        self.Vist_T[j_index][4] = self.Vist_T[j_index][2]

    def calculate_energy(self, time, drone_route, demand):     # 传入参数 无人机起飞节点的出发时间 无人机路径
        curent_load=demand
        arrival_time=0
        depart_time=0
        energy_neeed=0
        for i in range(1 , len(drone_route)):
            if i-1==0:
                prev_indices = drone_route[0]-1
                current_indices =  drone_route[i]-1
                travel_time  = self.ALLdistanceDmatrix[prev_indices+1][current_indices+1]/self.drone_speed
                arrival_time = travel_time+time
                energy_neeed =(curent_load+self.drone_weight)*travel_time*self.energy_fight
                wait_time=max(0, self.customers[current_indices].start_time-arrival_time)
                depart_time=arrival_time+wait_time+self.service_time
                energy_neeed += (curent_load + self.drone_weight) * wait_time * self.energy_hover
                energy_neeed += (curent_load + self.drone_weight) * self.service_time * self.energy_service
                customer = self.customers[current_indices]
                if customer.success is not False:  # None或True都视为成功（初始状态）
                    if customer.demand > 0:
                        curent_load -= customer.demand
                    else:
                        curent_load += abs(customer.demand)
            elif i == len(drone_route)-1:
                prev_indices =  drone_route[i-1]-1
                current_indices =  drone_route[i]-1
                travel_time = self.ALLdistanceDmatrix[prev_indices+1][current_indices+1] / self.drone_speed
                arrival_time += travel_time + depart_time
                energy_neeed += (curent_load + self.drone_weight) * travel_time * self.energy_fight
                wait_time = max(0, self.Vist_T[current_indices][2] - arrival_time)
                energy_neeed += (curent_load + self.drone_weight) * wait_time * self.energy_hover
            else:
                prev_indices =  drone_route[i - 1]-1
                current_indices =  drone_route[i]-1
                travel_time = self.ALLdistanceDmatrix[prev_indices+1][current_indices+1] / self.drone_speed
                arrival_time += travel_time + depart_time
                energy_neeed += (curent_load + self.drone_weight) * travel_time * self.energy_fight
                wait_time = max(0, self.customers[current_indices].start_time-arrival_time)
                depart_time = arrival_time + wait_time + self.service_time
                energy_neeed += (curent_load + self.drone_weight) * wait_time * self.energy_hover
                energy_neeed += (curent_load + self.drone_weight) * self.service_time * self.energy_service
                customer = self.customers[current_indices]
                if customer.success is not False:
                    if customer.demand > 0:
                        curent_load -= customer.demand
                    else:
                        curent_load += abs(customer.demand)
        return  energy_neeed

    def cost_single_vehicle(self, vehicle_id):                   # 计算单一车辆对成本
        cost=22.0  #固定成本
        lenth_truck=len(self.TRUCK_Routes[vehicle_id].Troute)
        for i in range(1, lenth_truck):
            curent_indices =  self.TRUCK_Routes[vehicle_id].Troute[i]
            if i == 1 or i == lenth_truck-1:
                cost += self.ALLdistanceTmatrix[0][curent_indices]*self.cost_truck
            else:
                prev_indices= self.TRUCK_Routes[vehicle_id].Troute[i-1]
                cost += self.ALLdistanceTmatrix[prev_indices][curent_indices] * self.cost_truck
        for trip in self.DRONE_Routes[vehicle_id].route:
            first_indices= trip['launch_node']-1
            energy=self.calculate_energy(self.Vist_T[first_indices][4], trip['path'], trip['initial_load'])
            cost +=energy*self.cost_drone
        return cost

    def cost(self):  # 计算所有成本
        cost=0.0
        lenth=len(self.TRUCK_Routes)
        for vehicle_id in range(lenth):
            cost+=self.cost_single_vehicle(vehicle_id)
        return cost

    # ==================== 信息素机制方法 ====================

    def initialize_pheromone_matrix(self):
        """
        初始化信息素矩阵
        矩阵大小为 (n+1) × (n+1)，包含仓库节点（索引0）
        """
        matrix_size = self.cnum + 1  # 客户数量 + 仓库
        self.pheromone_matrix = np.full((matrix_size, matrix_size), self.pheromone_initial, dtype=float)
        # 设置对角线为0（避免自循环）
        np.fill_diagonal(self.pheromone_matrix, 0.0)
        print(f"信息素矩阵初始化完成，大小: {matrix_size}×{matrix_size}")
        print(f"对角线检查: {np.sum(np.diag(self.pheromone_matrix))}")  # 应该为0

    def safe_node_index(self, node_id):
        """
        安全地处理节点索引，确保类型一致性
        """
        if node_id is None:
            return 0
        # 处理numpy类型
        if hasattr(node_id, 'item'):
            node_id = node_id.item()
        # 确保为整数
        node_id = int(node_id)
        # 确保在有效范围内
        if node_id < 0 or node_id > self.cnum:
            print(f"  警告：节点ID {node_id} 超出范围 [0, {self.cnum}]")
            return 0
        return node_id

    def update_pheromone(self, current_cost, new_cost):
        """
        基于解的质量改进更新信息素
        Args:
            current_cost: 当前解的成本
            new_cost: 新解的成本
        """
        if new_cost >= current_cost:  # 没有改进，不更新信息素
            return
        # 计算改进比例
        improvement_ratio = (current_cost - new_cost) / current_cost
        pheromone_increment = improvement_ratio * self.pheromone_learning_rate
        print(f" 更新信息素，改进比例: {improvement_ratio:.4f}")
        # 保存对角线状态用于检查
        diagonal_before = np.sum(np.diag(self.pheromone_matrix))
        # 更新卡车路径的信息素
        for truck_idx, truck in enumerate(self.TRUCK_Routes):
            route = truck.Troute
            print(f"   处理卡车{truck_idx}路径: {len(route)}个节点")
            for i in range(len(route) - 1):
                from_node = self.safe_node_index(route[i])
                to_node = self.safe_node_index(route[i + 1])
                #  避免更新对角线元素
                if from_node == to_node:
                    print(f" 跳过自循环: {from_node} -> {to_node}")
                    continue
                # 信息素更新公式
                old_pheromone = self.pheromone_matrix[from_node, to_node]
                new_pheromone = old_pheromone * (1 - self.pheromone_learning_rate) + pheromone_increment
                # 限制信息素范围
                self.pheromone_matrix[from_node, to_node] = np.clip(new_pheromone,
                                                                    self.pheromone_min,
                                                                    self.pheromone_max)
        # 更新无人机路径的信息素
        for drone_idx, drone in enumerate(self.DRONE_Routes):
            print(f"   处理无人机{drone_idx}路径: {len(drone.route)}个任务")
            for trip in drone.route:
                path = trip['path']
                for i in range(len(path) - 1):
                    from_node = self.safe_node_index(path[i])
                    to_node = self.safe_node_index(path[i + 1])
                    # 避免更新对角线元素
                    if from_node == to_node:
                        print(f"跳过自循环: {from_node} -> {to_node}")
                        continue
                    old_pheromone = self.pheromone_matrix[from_node, to_node]
                    new_pheromone = old_pheromone * (1 - self.pheromone_learning_rate) + pheromone_increment

                    self.pheromone_matrix[from_node, to_node] = np.clip(new_pheromone,
                                                                        self.pheromone_min,
                                                                        self.pheromone_max)
        #强制确保对角线为0
        np.fill_diagonal(self.pheromone_matrix, 0.0)
        # 验证对角线状态
        diagonal_after = np.sum(np.diag(self.pheromone_matrix))
        if abs(diagonal_after) > 1e-10:
            print(f" 警告：对角线不为0！更新前: {diagonal_before:.6f}, 更新后: {diagonal_after:.6f}")
        else:
            print(f" 对角线检查通过: {diagonal_after:.6f}")
        print(f" 信息素更新完成，改进比例: {improvement_ratio:.4f}")

    def evaporate_pheromone(self):
        """
        信息素挥发机制，防止过早收敛
        """
        # 保存对角线状态
        diagonal_before = np.sum(np.diag(self.pheromone_matrix))
        # 对所有信息素值进行挥发: p_{n,m} ← (1-ρ) · p_{n,m}
        self.pheromone_matrix *= (1 - self.pheromone_evaporation_rate)
        #强制确保对角线为0
        np.fill_diagonal(self.pheromone_matrix, 0.0)
        # 验证对角线状态
        diagonal_after = np.sum(np.diag(self.pheromone_matrix))
        if abs(diagonal_after) > 1e-10:
            print(f" 挥发后对角线异常！挥发前: {diagonal_before:.6f}, 挥发后: {diagonal_after:.6f}")
        print(f" 信息素挥发完成，挥发率: {self.pheromone_evaporation_rate}")

    def get_pheromone_guided_insertion_score(self, customer_id, prev_customer, next_customer,
                                             insertion_cost):
        """
        计算信息素指导的插入得分
        Args:
            customer_id: 要插入的客户ID
            prev_customer: 前一个客户ID
            next_customer: 后一个客户ID
            insertion_cost: 插入成本
        Returns:
            综合得分
        """
        # 安全处理节点索引
        customer_id = self.safe_node_index(customer_id)
        prev_id = self.safe_node_index(prev_customer)
        next_id = self.safe_node_index(next_customer)
        # 计算距离成分（倒数，距离越小分数越高）
        distance_score = 1.0 / max(insertion_cost, 0.1)  # 避免除零
        # 计算信息素成分
        pheromone_to_customer = self.pheromone_matrix[prev_id, customer_id]
        pheromone_from_customer = self.pheromone_matrix[customer_id, next_id]
        pheromone_score = pheromone_to_customer + pheromone_from_customer
        # 综合评分: α·距离分数 + β·信息素分数
        total_score = (self.pheromone_alpha * distance_score +
                       self.pheromone_beta * pheromone_score)
        return total_score

    def print_pheromone_info(self):
            """
            打印信息素相关信息
            """
            non_zero_pheromones = self.pheromone_matrix[self.pheromone_matrix > 0]
            if len(non_zero_pheromones) > 0:
                mean_val = np.mean(non_zero_pheromones)
                std_val = np.std(non_zero_pheromones)
                min_val = np.min(non_zero_pheromones)
                max_val = np.max(non_zero_pheromones)
                strong_connections = np.sum(self.pheromone_matrix > self.pheromone_initial * 2)
                weak_connections = np.sum((self.pheromone_matrix > 0) &
                                          (self.pheromone_matrix < self.pheromone_initial * 0.5))
                print("\n" + "=" * 40)
                print("信息素矩阵统计信息:")
                print(f"  平均值: {mean_val:.4f}")
                print(f"  标准差: {std_val:.4f}")
                print(f"  最小值: {min_val:.4f}")
                print(f"  最大值: {max_val:.4f}")
                print(f"  强连接数: {strong_connections}")
                print(f"  弱连接数: {weak_connections}")
                print("=" * 40 + "\n")

    def clean_route_data_types(self):
        """
        清理路径中的数据类型混合问题
        将所有numpy类型转换为标准Python int类型
        """
        print(" 开始清理数据类型...")
        # 清理卡车路径
        for truck_idx, truck in enumerate(self.TRUCK_Routes):
            original_route = truck.Troute.copy()
            cleaned_route = []
            for node in truck.Troute:
                # 处理numpy类型
                if hasattr(node, 'item'):
                    cleaned_node = int(node.item())
                else:
                    cleaned_node = int(node)
                cleaned_route.append(cleaned_node)

            truck.Troute = cleaned_route

            if original_route != cleaned_route:
                print(f"    卡车{truck_idx}路径已清理")
        # 清理无人机路径
        for drone_idx, drone in enumerate(self.DRONE_Routes):
            trips_cleaned = 0
            for trip in drone.route:
                original_path = trip['path'].copy()
                cleaned_path = []
                for node in trip['path']:
                    # 处理numpy类型
                    if hasattr(node, 'item'):
                        cleaned_node = int(node.item())
                    else:
                        cleaned_node = int(node)
                    cleaned_path.append(cleaned_node)
                trip['path'] = cleaned_path
                # 同时清理launch_node和retrieval_node
                if hasattr(trip['launch_node'], 'item'):
                    trip['launch_node'] = int(trip['launch_node'].item())
                else:
                    trip['launch_node'] = int(trip['launch_node'])
                if hasattr(trip['retrieval_node'], 'item'):
                    trip['retrieval_node'] = int(trip['retrieval_node'].item())
                else:
                    trip['retrieval_node'] = int(trip['retrieval_node'])
                if original_path != cleaned_path:
                    trips_cleaned += 1
            if trips_cleaned > 0:
                print(f"    无人机{drone_idx}: {trips_cleaned}个任务已清理")
        print(" 数据类型清理完成")

    # ==================== 局部搜索核心算法 ====================

    def local_search(self, truck_id: int, current_cost: float) -> float:
        """
        车辆对内局部搜索主控制器
        """
        if not self.validate_customer_assignment(truck_id, next(iter(self.get_vehicle_customers(truck_id)), -1)):
            return current_cost

        intra_operators = ['intra_move', 'intra_swap', 'intra_2opt']
        no_improve_count = 0
        best_cost = current_cost

        print(f"   🔄 开始车辆对{truck_id}局部搜索 (初始成本: {current_cost:.2f})")

        while no_improve_count < self.local_search_max_no_improve:
            selected_operator = random.choice(intra_operators)

            try:
                operation_success, new_cost = self._apply_intra_operator_with_feasibility_check(
                    truck_id, selected_operator, best_cost)

                if operation_success and new_cost < best_cost:
                    improvement = best_cost - new_cost
                    best_cost = new_cost
                    no_improve_count = 0
                    print(f"     ✅ {selected_operator}改进: {improvement:.2f}")
                else:
                    no_improve_count += 1

            except Exception as e:
                print(f"     ⚠️ {selected_operator}执行异常: {e}")
                no_improve_count += 1

        total_improvement = current_cost - best_cost
        if total_improvement > 0:
            print(f"   🎯 局部搜索完成，总改进: {total_improvement:.2f}")
        else:
            print(f"   ⚪ 局部搜索完成，无改进")

        return best_cost

    def enhanced_local_search_trigger(self, truck_id, cost_before, cost_after):
        """
        更宽松的局部搜索触发机制
        """
        should_trigger = False
        trigger_reason = ""
        # 添加调试日志
        print(f"      🔍 局部搜索触发检查: 成本 {cost_before:.2f} → {cost_after:.2f}")
        print(f"         操作计数: {self.ls_operation_count}")

        # 策略1: 成本改进或保持
        if cost_after <= cost_before:
            should_trigger = True
            trigger_reason = "成本改进/保持"
            print(f"         ✅ 触发原因: {trigger_reason}")

        # 策略2: 宽松阈值 (30%)
        elif cost_after <= cost_before * 1.30:  # 30%阈值
            should_trigger = True
            trigger_reason = "满足30%阈值"
            print(f"         ✅ 触发原因: {trigger_reason}")

        # 策略3: 频率触发 (每2次操作)
        self.ls_operation_count += 1
        if not should_trigger and self.ls_operation_count % 2 == 0:
            should_trigger = True
            trigger_reason = f"频率触发(第{self.ls_operation_count}次)"
            print(f"         ✅ 触发原因: {trigger_reason}")

        # 策略4: 强制触发 (每3次操作必须触发)
        if not should_trigger and self.ls_operation_count % 3 == 0:
            should_trigger = True
            trigger_reason = "强制触发"
            print(f"         ✅ 触发原因: {trigger_reason}")

        # 策略5: 最终保险 (操作计数大于5时强制触发)
        if not should_trigger and self.ls_operation_count > 5:
            should_trigger = True
            trigger_reason = "保险强制触发"
            print(f"         ✅ 触发原因: {trigger_reason}")

        if not should_trigger:
            print(f"         ❌ 未触发，继续等待")

        return should_trigger, trigger_reason


    def _apply_intra_operator_with_feasibility_check(self, truck_id: int, operator_name: str, current_cost: float):
        """
        执行车辆对内算子并进行可行性检查
        """
        # 备份当前状态
        backup_truck_route = copy.deepcopy(self.TRUCK_Routes[truck_id])
        backup_drone_routes = copy.deepcopy(self.DRONE_Routes[truck_id])

        try:
            # 执行局部搜索操作
            if operator_name == 'intra_move':
                operation_success = self._intra_move_within_vehicle(truck_id)
            elif operator_name == 'intra_swap':
                operation_success = self._intra_swap_within_vehicle(truck_id)
            elif operator_name == 'intra_2opt':
                operation_success = self._intra_2opt_within_vehicle(truck_id)
            else:
                return False, current_cost

            if not operation_success:
                return False, current_cost

            # 可行性检查和修复
            if hasattr(self, 'feasibility_repair_ops') and self.feasibility_repair_ops:
                feasible = self.feasibility_repair_ops.check_and_repair_feasibility(truck_id)
                if feasible:
                    new_cost = self.cost()
                    return True, new_cost

            # 如果没有可行性检查模块，直接计算成本
            new_cost = self.cost()
            return True, new_cost

        except Exception as e:
            print(f"       操作执行异常: {e}")
            return False, current_cost
        finally:
            # 如果操作失败或不可行，恢复备份状态
            if not hasattr(self, '_operation_successful') or not self._operation_successful:
                self.TRUCK_Routes[truck_id] = backup_truck_route
                self.DRONE_Routes[truck_id] = backup_drone_routes

    def _intra_move_within_vehicle(self, truck_id: int) -> bool:
        """车辆对内客户移动"""
        try:
            vehicle_customers = list(self.get_vehicle_customers(truck_id))
            if len(vehicle_customers) < 2:
                return False

            # 随机选择一个客户进行移动
            customer_to_move = random.choice(vehicle_customers)

            # 尝试在卡车路径内移动
            truck_route = self.TRUCK_Routes[truck_id].Troute
            if customer_to_move in truck_route and len(truck_route) > 3:
                current_pos = truck_route.index(customer_to_move)
                # 随机选择新位置（排除当前位置和仓库位置）
                valid_positions = [i for i in range(1, len(truck_route) - 1) if i != current_pos]
                if valid_positions:
                    new_pos = random.choice(valid_positions)
                    truck_route.remove(customer_to_move)
                    truck_route.insert(new_pos, customer_to_move)
                    self._update_customer_service_info(truck_id, customer_to_move)
                    return True

            return False

        except Exception as e:
            print(f"         intra_move异常: {e}")
            return False

    def _intra_swap_within_vehicle(self, truck_id: int) -> bool:
        """车辆对内客户交换"""
        try:
            truck_route = self.TRUCK_Routes[truck_id].Troute
            customer_positions = list(range(1, len(truck_route) - 1))  # 排除仓库位置

            if len(customer_positions) < 2:
                return False

            # 随机选择两个位置进行交换
            pos1, pos2 = random.sample(customer_positions, 2)
            truck_route[pos1], truck_route[pos2] = truck_route[pos2], truck_route[pos1]

            # 更新客户服务信息
            self._update_customer_service_info(truck_id, truck_route[pos1])
            self._update_customer_service_info(truck_id, truck_route[pos2])

            return True

        except Exception as e:
            print(f"         intra_swap异常: {e}")
            return False

    def _intra_2opt_within_vehicle(self, truck_id: int) -> bool:
        """车辆对内2-opt改进"""
        try:
            truck_route = self.TRUCK_Routes[truck_id].Troute
            if len(truck_route) < 5:  # 至少需要3个客户
                return False

            # 随机选择两个边进行2-opt交换
            n = len(truck_route) - 2  # 排除仓库
            if n < 3:
                return False

            i = random.randint(1, n - 2)  # 第一个边的起点
            j = random.randint(i + 2, n)  # 第二个边的起点

            # 执行2-opt交换：逆转i到j之间的路径
            truck_route[i:j + 1] = truck_route[i:j + 1][::-1]

            # 更新相关客户的服务信息
            for pos in range(i, j + 1):
                if pos < len(truck_route):
                    self._update_customer_service_info(truck_id, truck_route[pos])

            return True

        except Exception as e:
            print(f"         intra_2opt异常: {e}")
            return False

    def _update_customer_service_info(self, truck_id: int, customer_id: int):
        """更新客户服务信息"""
        try:
            if customer_id <= len(self.customers):
                customer = self.customers[customer_id - 1]
                # 确保客户的service_by信息正确指向当前车辆对
                customer.service_by = ["tk", truck_id]
        except Exception as e:
            print(f"         更新客户服务信息异常: {e}")