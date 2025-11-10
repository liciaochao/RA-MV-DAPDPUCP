import math
import copy
from typing import List, Dict, Tuple, Optional


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

        # 🆕 改进的修复参数
        self.max_repair_attempts = 3  # 大幅减少最大尝试次数
        self.violation_tolerance = 0.1  # 违反容忍度
        self.enable_aggressive_repair = True  # 启用激进修复模式
        self.debug_mode = False  # 调试模式开关

    def check_and_repair_feasibility(self, truck_id: int) -> bool:
        """
        🆕 改进版可行性检查和修复 - 彻底解决无限循环问题
        """
        if self.debug_mode:
            print(f"🔧 开始车辆对{truck_id}可行性检查（改进版）...")

        repair_attempts = 0
        max_attempts = self.max_repair_attempts
        overall_success = True

        # 🆕 关键改进：记录已处理的违反类型，防止无限循环
        processed_violations = set()
        consecutive_same_violations = 0
        last_violation_signature = None

        while repair_attempts < max_attempts:
            violations_found = False

            # 🆕 一次性检查所有违反类型
            violation_summary = self._comprehensive_violation_check(truck_id)

            if not violation_summary:
                if self.debug_mode:
                    print(f"   ✅ 车辆对{truck_id}所有约束都已满足")
                break

            # 🆕 生成违反签名，检测是否陷入循环
            current_signature = self._generate_violation_signature(violation_summary)
            if current_signature == last_violation_signature:
                consecutive_same_violations += 1
                if consecutive_same_violations >= 2:  # 连续2次相同违反就跳出
                    if self.debug_mode:
                        print(f"   ⚠️ 检测到循环违反，启动激进修复...")
                    break
            else:
                consecutive_same_violations = 0

            last_violation_signature = current_signature

            # 🆕 按优先级处理违反（一次只处理一种类型）
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
                            print(f"     ✅ {violation_type}修复成功，重新计算状态...")
                        # 🆕 关键：修复后立即重新计算所有状态
                        self._recalculate_all_states(truck_id)
                    else:
                        if self.debug_mode:
                            print(f"     ❌ {violation_type}修复失败")
                        overall_success = False

                    # 🆕 每次修复后立即跳出，重新检查
                    break

            if not repair_performed:
                if self.debug_mode:
                    print(f"   ⚠️ 无新的违反需要处理")
                break

            repair_attempts += 1

        # 🆕 如果常规修复失败或检测到循环，启动激进修复
        if (repair_attempts >= max_attempts or consecutive_same_violations >= 2) and self.enable_aggressive_repair:
            if self.debug_mode:
                print(f"   🚨 启动激进修复模式...")
            aggressive_success = self._aggressive_repair_mode(truck_id)
            if aggressive_success:
                overall_success = True
                if self.debug_mode:
                    print(f"   ✅ 激进修复成功")
            else:
                overall_success = False
                if self.debug_mode:
                    print(f"   ❌ 激进修复失败")

        return overall_success

    def _generate_violation_signature(self, violation_summary: Dict) -> str:
        """🆕 生成违反签名，用于检测循环"""
        signature_parts = []
        for violation_type, violations in violation_summary.items():
            if violations:
                signature_parts.append(f"{violation_type}:{len(violations)}")
        return "|".join(sorted(signature_parts))

    def _comprehensive_violation_check(self, truck_id: int) -> Dict:
        """🆕 全面的违反检查 - 一次性检查所有类型"""
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
                print(f"     ❌ 违反检查出错: {e}")

        return violations

    def _check_truck_load_violations_detailed(self, truck_id: int) -> List[Dict]:
        """🆕 详细的卡车载重检查"""
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
        """🆕 详细的无人机载重检查"""
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
        """🆕 检查飞行过程中的载重变化"""
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
        """🆕 详细的能耗检查"""
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
        """🆕 详细的时间窗口检查 - 只检查显著违反"""
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
        """🆕 根据违反类型执行对应的修复"""
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
        """🆕 修复卡车载重违反"""
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
        """🆕 修复无人机载重违反"""
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
        """🆕 修复起飞过载"""
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
        """🆕 修复飞行中过载"""
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
        """🆕 修复无人机能耗违反"""
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
        """🆕 改进的时间窗口修复 - 减少不必要的等待时间设置"""
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
        """🆕 将卡车客户移到更早位置"""
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
        """🆕 将无人机客户移到更早位置"""
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
        """🆕 添加无人机悬停能耗"""
        try:
            if trip_idx is not None and trip_idx < len(self.dyn_opt.DRONE_Routes[truck_id].route):
                trip = self.dyn_opt.DRONE_Routes[truck_id].route[trip_idx]
                additional_energy = wait_time * self.dyn_opt.energy_hover * self.dyn_opt.drone_weight
                trip['energy'] = trip.get('energy', 0) + additional_energy
        except Exception as e:
            if self.debug_mode:
                print(f"       添加悬停能耗出错: {e}")

    def _aggressive_repair_mode(self, truck_id: int) -> bool:
        """🆕 激进修复模式 - 最后的修复手段"""
        if self.debug_mode:
            print(f"   🚨 执行车辆对{truck_id}激进修复...")

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
                print(f"     ✅ 激进修复完成")
            return True

        except Exception as e:
            if self.debug_mode:
                print(f"     ❌ 激进修复失败: {e}")
            return False

    def _recalculate_all_states(self, truck_id: int):
        """🆕 重新计算车辆所有状态 - 关键的状态同步方法"""
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