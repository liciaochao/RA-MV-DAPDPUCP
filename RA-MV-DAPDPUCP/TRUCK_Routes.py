import math
class TRUCKtsp:
    def __init__(self, customers,  distance_matrix,  vehicle_speed, max_return_time, service_time, wait_time_weight, truck, ALLcustomers, drone_weight, start_node=0):
        """
        初始化优化的最近邻TSP求解器
        :param customers:           客户数组   [客户编号, x坐标, y坐标, 需求, 最早开始服务时间, 最晚开始服务时间,是否接受无人机服务]
        :param distance_matrix:     客户之间的距离矩阵，形状为 (n, n)
        :param vehicle_speed:       车辆速度（单位：距离/小时）
        :param max_return_time:     返回起点的最晚时间
        :param service_time:        每个客户的服务时间（单位：分钟）
        :param wait_time_weight:    等待时间权重，用于调整距离和等待时间的平衡
        :param truck:               服务的车辆
        :param ALLcustomers:        存储所有客户实例的列表
        :param start_node:          起始节点索引（默认为0，即第一个客户）
        """
        self.customers = customers
        self.distance_matrix = distance_matrix
        self.vehicle_speed = vehicle_speed
        self.max_return_time = max_return_time
        self.service_time = service_time
        self.wait_time_weight = wait_time_weight    # 调整等待时间的权重
        self.truck = truck
        self.Acus = ALLcustomers
        self.drone_weight=drone_weight
        self.start_node = start_node
        self.num_customers = len(customers)
        self.route = []                             # 存储最终求解的路径
        self.total_distance = 0                     # 总旅行距离
        self.total_time = 0                         # 总行驶时间

    def solve(self):
        """
        使用优化的最近邻算法求解TSP问题，考虑等待时间、最早服务时间和最晚服务时间
        """
        total_demand = sum(self.customers[idx][3] for idx in range(self.num_customers) if self.customers[idx][3] > 0)      #卡车服务送货客户的总容量
        remaining_demand=self.truck.max_capacity-total_demand-self.drone_weight                                            #剩余卡车容量
        visited = [False] * self.num_customers  # 记录客户是否已访问
        self.route = [self.start_node]          # 起始客户加入路径
        visited[self.start_node] = True         # 标记起始客户为已访问
        self.total_time = 0                     # 重置总时间
        self.total_distance = 0                 # 重置总距离

        current_customer = self.start_node
        remaining_customers = set(range(self.num_customers)) - {self.start_node}    # 剩余未访问的客户

        while remaining_customers:
            best_customer = None
            best_cost = float('inf')
            for next_customer in list(remaining_customers):
                if self.customers[next_customer][3]<0:                              # 判断是否为取货客户
                    if abs(self.customers[next_customer][3])>remaining_demand:      # 进一步判断能否为取货客户服务
                        continue
                # 计算到下一个客户的行驶时间
                travel_time = self.distance_matrix[current_customer][next_customer] / self.vehicle_speed
                # 估算到达该客户的时间（当前时间 + 行驶时间）
                arrival_time = self.total_time + travel_time
                # 计算等待时间（如果早于最早服务时间则需要等待）
                if arrival_time > self.customers[next_customer][5]:
                    continue  # 如果离开时间超过最晚服务时间，则跳过该客户
                wait_time = max(0, self.customers[next_customer][4] - arrival_time)
                # 完成服务后的离开时间
                departure_time = arrival_time + wait_time + self.service_time
                # 计算从下一个客户返回起点的时间
                return_to_start_time = departure_time + (self.distance_matrix[next_customer][self.start_node] / self.vehicle_speed)
                # 判断是否满足时间约束条件
                if return_to_start_time <= self.max_return_time:
                    # 计算综合成本（距离 + 等待时间 * 权重）
                    cost = self.distance_matrix[current_customer][next_customer] + self.wait_time_weight * wait_time
                    # 更新最优客户
                    if cost < best_cost:
                        best_customer = next_customer
                        best_cost = cost

            # 如果找不到满足条件的客户，跳出循环
            if best_customer is None:
                print("无法找到满足约束条件的路径，执行回溯")
                break  # 退出循环
            #卡车到达时间
            self.Acus[self.customers[best_customer][0] - 1].arrive_truck = self.total_time +self.distance_matrix[current_customer][ best_customer] / self.vehicle_speed
            # 无人机到达时间
            self.Acus[self.customers[best_customer][0] - 1].arrive_drone = self.Acus[self.customers[best_customer][0] - 1].arrive_truck
            # 服务等待时间
            self.Acus[self.customers[best_customer][0] - 1].wait_time = max(0, self.customers[best_customer][4] -self.Acus[self.customers[best_customer][0] - 1].arrive_truck)
            # 服务开始时间
            self.Acus[self.customers[best_customer][0] - 1].service_begin = self.Acus[self.customers[best_customer][0] - 1].arrive_truck + self.Acus[self.customers[best_customer][0] - 1].wait_time
            # 卡车离开时间
            self.Acus[self.customers[best_customer][0] - 1].departure_truck = self.Acus[self.customers[best_customer][0] - 1].service_begin + self.service_time
            # 无人机离开时间
            self.Acus[self.customers[best_customer][0] - 1].departure_drone = self.Acus[self.customers[best_customer][0] - 1].departure_truck
            # 标记客户由哪一个卡车或无人机服务
            self.Acus[self.customers[best_customer][0] - 1].service_by = ["tk", self.truck.vehicle_id]
            # 更新路径和状态
            self.route.append(best_customer)
            self.total_distance += self.distance_matrix[current_customer][best_customer]
            self.total_time += self.distance_matrix[current_customer][best_customer] / self.vehicle_speed + wait_time + self.service_time
            visited[best_customer] = True
            current_customer = best_customer
            remaining_customers.remove(best_customer)  # 从剩余客户中移除已访问的客户
            if self.customers[best_customer][3] < 0:  # 判断是否为取货客户
                remaining_demand-=self.customers[best_customer][3]
                remaining_demand=max(0, remaining_demand)
            else:
                remaining_demand+=self.customers[best_customer][3]
                remaining_demand = max(self.truck.max_capacity, remaining_demand)
        # 确保所有客户都已访问后，返回起点以闭合路径
        if len(self.route) == self.num_customers:
            self.route.append(self.start_node)
            self.total_distance += self.distance_matrix[current_customer][self.start_node]
            self.total_time += self.distance_matrix[current_customer][self.start_node] / self.vehicle_speed
            self.truck.end_time=self.total_time
        else:
            print("无法访问所有客户节点")

    def get_route(self):
        """
        获取计算得到的旅行路径
        :return: 按客户编号返回访问顺序
        """
        route=[int(self.customers[i][0]) for i in self.route]
        return route  # 使用 int() 转换为标准的 Python 整型

    def get_total_distance(self):
        """
        获取计算得到的总旅行距离
        :return: 总旅行距离
        """
        return self.total_distance

    def get_total_time(self):
        """
        获取计算得到的总旅行时间
        :return: 总旅行时间
        """
        return self.total_time