import numpy as np
import copy
import math
class AddDroneRoute:
    def __init__(self, customers, distance_matrix, distanceDmatrix, truck_speed, drone_speed, drone_weight, max_capacity, max_battery, service_time, drone, truck , allcustomers, energy_fight, energy_service, energy_hover):
        """
        初始化优化的最近邻TSP求解器
        :param customers:           客户数组   [客户编号, x坐标, y坐标, 需求, 最早开始服务时间, 最晚开始服务时间,是否接受无人机服务]
        :param distance_matrix:     客户之间的距离矩阵，卡车路径
        :param distanceDmatrix:     客户之间的距离矩阵，无人机路径
        :param drone_speed:         无人机速度（单位：米/分钟）
        :param max_capacity:        无人机载重
        :param max_battery:         无人机电池最大能量
        :param service_time:        单位客户服务时间
        :param drone:               服务的无人机
        :param truck:               服务的卡车
        :param aLLcustomers:        存储所有客户实例的列表
        :param energy_fight:        飞行时的能量消耗系数
        :param energy_service:      服务时的能量消耗系数
        :param energy_hover:        悬浮时的能量消耗系数
        """
        self.customers = customers
        self.distance_matrix = distance_matrix
        self.distanceDmatrix = distanceDmatrix
        self.truck_speed=truck_speed
        self.drone_speed = drone_speed
        self.drone_weight= drone_weight
        self.max_capacity = max_capacity
        self.max_battery = max_battery
        self.service_time = service_time
        self.drone = drone
        self.truck = truck
        self.allcustomers = allcustomers
        self.energy_fight = energy_fight
        self.energy_service = energy_service
        self.energy_hover = energy_hover
        self.cnum = customers.shape[0]                                          # 客户数量
        self.T=self.Initial_T()                                                 # 时间矩阵  仅包含客户
        self.potential_customers = customers[customers[:, -1] == 1]             # 将能够接收无人机配送的客户选出
        # 计算每个客户的绕行距离，并按降序排序
        indices = np.argsort([self.calculate_detour_distance(customer) for customer in self.potential_customers])[::-1]
        # 根据排序的索引重新排列 self.potential_customers
        self.potential_customers = self.potential_customers[indices]

        # 初始化时间矩阵 T
    def Initial_T(self):
        #dtype='object'：这里我们将 T 数组初始化为 object 类型，这样它就能接受任意类型的数据（整数和浮动数）。这可以确保在赋值过程中不会发生类型冲突。
        T = np.empty((self.cnum - 1, 5), dtype='object')  # 用 object 类型初始化，避免重复赋值
        # 填充数据
        for i in range(1, self.cnum):
            T[i - 1, 0] = self.customers[i][0]  # 第一列为整数类型
        for j in range(1, self.cnum):  # 卡车到达时间
            T[j - 1, 1] = self.allcustomers[self.customers[j][0] - 1].arrive_truck
        for k in range(1, self.cnum):  # 卡车离开时间
            T[k - 1, 2] = self.allcustomers[self.customers[k][0] - 1].departure_truck
        for m in range(1, self.cnum):  # 无人机到达时间
            T[m - 1, 3] = self.allcustomers[self.customers[m][0] - 1].arrive_drone
        for n in range(1, self.cnum):  # 无人机离开时间
            T[n - 1, 4] = self.allcustomers[self.customers[n][0] - 1].departure_drone
        # 打印结果
        print(T)
        return T

# 更新时间矩阵 T
    def Update_T(self, index_customer):
        if index_customer==len(self.truck.Troute)-1:                                #如果传入节点为路径中的最后一个节点 则不用更新 直接返回
            return
        launch_node=[trip['launch_node'] for trip in self.drone.route]              #起飞节点集合
        retrieval_node=[trip['retrieval_node'] for trip in self.drone.route]        #回收节点集合
        for j in range(index_customer, len(self.truck.Troute)-1):                     #从传入节点开始更新时间
            if j-1==0:                                                              #假设 是第一个客户节点
                j_index=np.where(self.customers[:, 0] == self.truck.Troute[j])[0][0]-1
                distance=self.distance_matrix[0][j_index+1]
                self.T[j_index][1]=distance/self.truck_speed                        #更新卡车到达时间
                self.T[j_index][2]=max(self.T[j_index][1], self.allcustomers[self.T[j_index][0]-1].start_time)+self.service_time    #更新卡车离开时间
                if self.truck.Troute[j] in launch_node :                                        #假设 当前节点仅为起飞节点时
                    self.T[j_index][3] = self.T[j_index][1]                                     #无人机
                    self.T[j_index][4] = self.T[j_index][1]
                    for trip in self.drone.route:                                           #更新以当前节点为起飞节点的无人机路径
                        if trip['launch_node'] == self.truck.Troute[j]:
                            trip=trip['path']
                            for i in range(1, len(trip)):
                                prev_indices = np.where(self.customers[:, 0] == trip[i - 1])[0][0]-1
                                current_indices = np.where(self.customers[:, 0] == trip[i])[0][0]-1
                                distance = self.distanceDmatrix[prev_indices+1][current_indices+1]
                                self.T[current_indices][3] = self.T[prev_indices][4]+distance/self.drone_speed
                                self.T[current_indices][4] = max(self.T[current_indices][3], self.allcustomers[self.T[current_indices][0]-1].start_time)+self.service_time
                                if trip[i] not in retrieval_node:
                                    self.T[current_indices][1] = 0
                                    self.T[current_indices][2] = 0
                else:
                    self.T[j_index][3] = self.T[j_index][1]
                    self.T[j_index][4] = self.T[j_index][2]
            else:
                j_index = np.where(self.customers[:, 0] == self.truck.Troute[j])[0][0]-1
                prev_indices = np.where(self.customers[:, 0] == self.truck.Troute[j-1])[0][0]-1
                distance = self.distance_matrix[prev_indices+1][j_index+1]
                self.T[j_index][1] = distance / self.truck_speed+self.T[prev_indices][2]
                self.T[j_index][2] = max(self.T[j_index][1], self.allcustomers[self.T[j_index][0] - 1].start_time) + self.service_time
                if self.T[j_index][3] == 0:                                     #当前节点 无人机
                    self.T[j_index][3] = 0
                    self.T[j_index][4] = 0
                else:
                    if self.truck.Troute[j] not in launch_node and self.truck.Troute[j] not in retrieval_node:       #当前节点为普通客户节点时
                        self.T[j_index][3] = self.T[j_index][1]
                        self.T[j_index][4] = self.T[j_index][2]
                    if self.truck.Troute[j] in launch_node and self.truck.Troute[j] not in retrieval_node:           #当前节点仅为起飞节点时
                        self.T[j_index][3] = self.T[j_index][1]
                        self.T[j_index][4] = self.T[j_index][1]
                        for trip in self.drone.route:
                            if trip['launch_node'] == self.truck.Troute[j]:
                                trip = trip['path']
                                for i in range(1, len(trip)):
                                    prev_indices = np.where(self.customers[:, 0] == trip[i - 1])[0][0]-1
                                    current_indices = np.where(self.customers[:, 0] == trip[i])[0][0]-1
                                    distance = self.distanceDmatrix[prev_indices+1][current_indices+1]
                                    self.T[current_indices][3] = self.T[prev_indices][4] + distance / self.drone_speed
                                    self.T[current_indices][4] = max(self.T[current_indices][3],
                                                       self.allcustomers[self.T[current_indices][0] - 1].start_time) + self.service_time
                                    if trip[i] not in retrieval_node:
                                        self.T[current_indices][1] = 0
                                        self.T[current_indices][2] = 0
                    if self.truck.Troute[j] in launch_node and self.truck.Troute[j] in retrieval_node:               #当前节点既为起飞节点又为回收节点时
                        max_time=max(self.T[j_index][2], self.T[j_index][3])
                        self.T[j_index][2] = max_time
                        self.T[j_index][4] = self.T[j_index][3]
                        for trip in self.drone.route:
                            if trip['launch_node'] == self.truck.Troute[j]:
                                trip = trip['path']
                                for i in range(1, len(trip)):
                                    prev_indices = np.where(self.customers[:, 0] == trip[i - 1])[0][0]-1
                                    current_indices = np.where(self.customers[:, 0] == trip[i])[0][0]-1
                                    distance = self.distanceDmatrix[prev_indices+1][current_indices+1]
                                    self.T[current_indices][3] = self.T[i - 1][4] + distance / self.drone_speed
                                    self.T[current_indices][4] = max(self.T[current_indices][3],
                                                       self.allcustomers[self.T[current_indices][0] - 1].start_time) + self.service_time
                                    if trip[i] not in retrieval_node:
                                        self.T[current_indices][1] = 0
                                        self.T[current_indices][2] = 0
                    if self.truck.Troute[j] not in launch_node and self.truck.Troute[j] in retrieval_node:  # 当前节点仅为回收节点时
                        max_time = max(self.T[j_index][2], self.T[j_index][3])
                        self.T[j_index][2] = max_time
                        self.T[j_index][4] = self.T[j_index][2]

    def calculate_detour_distance(self, customer):
        """
        计算客户插入路径后的绕行距离。假设客户插入位置不影响其他客户路径
        """
        # 获取客户在路径中的索引
        i = self.truck.Troute.index(customer[0])  # 使用 customer[0] 获取客户编号
        # 确保索引不超出路径范围
        if i == 1 or i == len(self.truck.Troute) - 2:
            return 0
        j = i - 1
        k = i + 1
        # 计算绕行距离
        dis_ji = self.distance_matrix[j][i]
        dis_ik = self.distance_matrix[i][k]
        return dis_ji + dis_ik

    def update_customer_information(self):
        launch_node = [trip['launch_node'] for trip in self.drone.route]  # 起飞节点集合
        retrieva_node = [trip['retrieval_node'] for trip in self.drone.route]  # 回收节点集合
        for i in range(self.cnum-1):
            if self.T[i][1] == 0:
                self.allcustomers[self.T[i][0] - 1].service_by=["de", self.drone.vehicle_id]
                self.allcustomers[self.T[i][0] - 1].service_begin=max(self.T[i][3], self.allcustomers[self.T[i][0] - 1].start_time)
                self.allcustomers[self.T[i][0] - 1].wait=max(0, self.allcustomers[self.T[i][0] - 1].start_time-self.T[i][3])
            else:
                self.allcustomers[self.T[i][0] - 1].service_begin = max(self.T[i][1], self.allcustomers[self.T[i][0] - 1].start_time)
                self.allcustomers[self.T[i][0] - 1].wait = max(0, self.allcustomers[self.T[i][0] - 1].start_time -self.T[i][1])
            self.allcustomers[self.T[i][0] - 1].arrive_truck = self.T[i][1]
            self.allcustomers[self.T[i][0] - 1].departure_truck = self.T[i][2]
            self.allcustomers[self.T[i][0] - 1].arrive_drone = self.T[i][3]
            self.allcustomers[self.T[i][0] - 1].departure_drone = self.T[i][4]
            if self.T[i][0] in launch_node:
                self.allcustomers[self.T[i][0] - 1].launch=1
            if self.T[i][0] in retrieva_node:
                self.allcustomers[self.T[i][0] - 1].retrieve=1

    def assign_customers_to_drone(self):
        """
        将客户分配给无人机，判断是否能够插入现有的无人机路径。
        """
        route=None
        modify=False                                                                    # 用于判断是否在原无人机路径中插入客户
        energy=0
        for customer in self.potential_customers:
            launch_node = [trip['launch_node'] for trip in self.drone.route]            # 起飞节点集合
            retrieva_node = [trip['retrieval_node'] for trip in self.drone.route]       # 回收节点集合
            truck_route_copy = copy.deepcopy(self.truck.Troute)                         # 深拷贝卡车路径
            drone_route_copy = copy.deepcopy(self.drone.route)                          # 深拷贝无人机路径
            T_copy = copy.deepcopy(self.T)                                              # 深拷贝时间矩阵
            if customer[0] in launch_node or customer[0] in retrieva_node:              # 如果该客户为发射节点或回收节点 则不移除
                continue
            if abs(customer[3])>self.max_capacity:                                      # 客户需求大于无人机容量 则不适用无人机
                continue
            if not self.drone.route:                                                    # 检查是否为空列表
                energy, route=self.AddNEWRoute(customer)                                # 目前还没有无人机路径，则创建新路径
            else:
                for index, existing_route in enumerate(self.drone.route):
                    demand=existing_route['initial_load']
                    existing_route = existing_route['path']

                    Existing_route = existing_route[1:-1]
                    total_demand_after_insertion = sum(self.allcustomers[c - 1].demand for c in Existing_route if
                                                       self.allcustomers[c - 1].demand > 0) + customer[3]
                    if total_demand_after_insertion > self.max_capacity:
                        continue
                    energy, route = self.AddIntoRoute(existing_route, customer, demand)
                    if route:                                       # 如果可以插入当前路径
                        self.drone.route[index]['path'] = route
                        self.drone.route[index]['energy'] = energy
                        modify = True                               # 标记修改成功
                        break                                       # 找到路径后可以停止遍历
                                                                    # 如果没有修改现有路径，则创建新路径
                if not modify:
                       energy, route = self.AddNEWRoute(customer)   # 创建新无人机路径
                                                                    # 如果找不到合适的路径插入客户，则跳过该客户
            if not route and not modify:                            # 如果无法插入 则还原卡车路径余时间矩阵
                self.truck.Troute = truck_route_copy                # 还原卡车路径
                self.drone.route =  drone_route_copy                # 还原无人机路径
                self.T = T_copy                                     # 还原时间矩阵
            if route and not modify:                                # 如果插入新的无人机路径中 则更新行程以及客户信息
                self.drone.add_trip(route[0],route[2], route, energy)
                self.update_customer_information()
                print(self.T)
            if route and modify:                                    # 如果插入已有的无人机路径中 则更新行程以及客户信息
                for k in range(1, len(route)):
                    prev_customer = route[k - 1]                    # 前一个客户
                    curent_customer = route[k]                      # 当前客户
                    prev_indices = np.where(self.customers[:, 0] == prev_customer)[0][0] - 1  # 找到其在 customers 数组中的行索引，便于计算距离
                    curent_indices = np.where(self.customers[:, 0] == curent_customer)[0][0] - 1
                    distance = self.distanceDmatrix[prev_indices + 1][curent_indices + 1]
                    arrival_time = self.T[prev_indices][4] + distance / self.drone_speed
                    wait_time = max(0, self.allcustomers[curent_customer - 1].start_time - arrival_time)
                    if k < len(route) - 1:
                        self.T[curent_indices][1] = 0
                        self.T[curent_indices][2] = 0
                        self.T[curent_indices][3] = arrival_time
                        self.T[curent_indices][4] = arrival_time + wait_time + self.service_time
                        # 更新在回收节点时无人机的离开时间
                    if k == len(route) - 1:
                        self.T[curent_indices][3] = arrival_time
                        if self.T[curent_indices][3] > self.T[curent_indices][2]:   # 假设 卡车需要等待无人机返回
                            self.T[curent_indices][2] = self.T[curent_indices][3]   # 更新 卡车的离开时间
                            if curent_customer in launch_node:                      # 假设 无人机返回后马上进行新的行程
                                self.T[curent_indices][4] = self.T[curent_indices][3]  # 更新 无人机离开时间
                            else:
                                self.T[curent_indices][4] = self.T[curent_indices][2]
                        elif self.T[curent_indices][3] < self.T[curent_indices][1]:  # 假设 无人机需要悬浮等待卡车
                            if curent_customer in launch_node:                       # 假设 无人机返回后马上进行新的行程
                                self.T[curent_indices][4] = self.T[curent_indices][1]  # 更新 无人机离开时间
                            else:
                                self.T[curent_indices][4] = self.T[curent_indices][2]
                        else:  # 假设 不需要等待
                            if curent_customer in launch_node:  # 假设 无人机返回后马上进行新的行程
                                self.T[curent_indices][4] = self.T[curent_indices][3]  # 更新 无人机离开时间
                            else:
                                self.T[curent_indices][4] = self.T[curent_indices][2]
                i_index = self.truck.Troute.index(route[len(route) - 1]) + 1
                self.update_customer_information()
                print(self.T)
                self.Update_T(i_index)
                print(self.T)
                self.update_customer_information()
                print(self.T)
            modify = False
            print(route)
        print(self.T)

    def AddNEWRoute(self, customer_i):
        """
        为客户 i 创建新的无人机路径，选择合适 的起飞节点与回收节点
        """
        launch_node = [trip['launch_node'] for trip in self.drone.route]            # 起飞节点集合
        retrieva_node = [trip['retrieval_node'] for trip in self.drone.route]       # 回收节点集合
        drone_route=[]
        # 1. 选择距离客户 i 最近的相邻节点作为起飞节点与回收节点
        i_index = self.truck.Troute.index(customer_i[0])                            # 使用 customer[0] 获取客户编号
        prev_index=i_index-1
        takeoff_node=self.truck.Troute[prev_index]
        latter_index=i_index+1
        retrieval_node=self.truck.Troute[latter_index]
        if i_index==1:              #如果是移除第一个客户节点
            prev_index = i_index + 1
            takeoff_node = self.truck.Troute[prev_index]
            latter_index = i_index + 2
            retrieval_node = self.truck.Troute[latter_index]
        if latter_index==len(self.truck.Troute)-1:
            return 0, []
        if self.T[prev_index][3]==0:                               # 若起飞节点上无无人机 则无法起飞
            return 0, []
        del self.truck.Troute[i_index]                             # 将客户从卡车路径中删除
        self.Update_T(i_index)                                     # 更新时间矩阵
        drone_route=[takeoff_node, customer_i[0], retrieval_node]  # 生成一条新的无人机路径
        # 2. 验证容量约束
        if abs(customer_i[3]) > self.max_capacity:
            return 0, []                                           # 超过容量约束，无法创建路径
        # 3. 验证时间约束
        prev_indices = np.where(self.customers[:, 0]   == takeoff_node)[0][0]
        curent_indices = np.where(self.customers[:, 0] == customer_i[0])[0][0]
        latter_indices = np.where(self.customers[:, 0] == retrieval_node)[0][0]
        distance=self.distanceDmatrix[prev_indices][curent_indices]
        time_travel =distance/self.drone_speed
        time_arrival=time_travel+self.T[prev_indices-1][1]
        if time_arrival>self.allcustomers[customer_i[0]-1].end_time:
            return 0, []                     # 违反时间窗约束，无法创建路径
        else:
            self.T[curent_indices-1][1] = 0
            self.T[curent_indices-1][2] = 0
            self.T[curent_indices-1][3] = time_arrival
            self.T[curent_indices-1][4] = max(time_arrival, self.allcustomers[customer_i[0]-1].start_time)+self.service_time
            self.T[latter_indices-1][3] = self.T[curent_indices-1][4]+self.distanceDmatrix[curent_indices][latter_indices]/self.drone_speed
            #如果返回过晚，则需要卡车等待
            if self.T[latter_indices-1][3]>self.T[latter_indices-1][2]:
                self.T[latter_indices-1][2]=self.T[latter_indices-1][3]
            # 如果回收节点也是起飞节点
            if retrieval_node in launch_node:
                self.T[latter_indices-1][4] = max(self.T[latter_indices-1][3],self.T[latter_indices-1][1])
            else:
                self.T[latter_indices-1][4] = self.T[latter_indices-1][2]
        self.Update_T(latter_index+1)
        #更新时间矩阵 验证插入节点到无人机路径以后的所有节点的时间约束是否得到满足
        for i in range(self.cnum-1):
            if self.T[i][1]==0:
                if self.T[i][3] > self.allcustomers[self.T[i][0]-1].end_time:
                    return 0, []
            elif self.T[i][3] == 0:
                if self.T[i][1] > self.allcustomers[self.T[i][0]-1].end_time:
                    return 0, []
            else:
                if self.T[i][1] > self.allcustomers[self.T[i][0]-1].end_time:
                    return 0, []
       # 4. 计算能量消耗（ 从起飞节点到客户 i，再到回收节点）
        total_energy_needed = self.calculate_energy(self.T[prev_indices-1][1], drone_route, customer_i[3])              #传入无人机在起飞节点出发时间  无人机路径
        if total_energy_needed > self.max_battery :
            return 0, []                                                                                                # 无法满足能量约束，返回空路径
        return total_energy_needed, drone_route

    def AddIntoRoute(self, route, customer_i, demand):
        """
        尝试将客户插入到现有路径中。如果可以插入，则返回True并更新路径，否则返回False
        """
        launch_node = [trip['launch_node'] for trip in self.drone.route]            # 起飞节点集合
        retrieval_node = [trip['retrieval_node'] for trip in self.drone.route]      # 回收节点集合
        need_delete_index = self.truck.Troute.index(customer_i[0])                  # 使用 customer[0] 获取客户编号 如果客户可以插入 方便以后删除客户
        modify = 0
        route_length = len(route)
        candidate_route = []
        first_indices = np.where(self.customers[:, 0] == route[0])[0][0] - 1                # 无人机路径上第一个节点的索引
        orignal_energy_needed = self.calculate_energy(self.T[first_indices][4], route, demand)
        total_demand=demand
        remaining_demand = self.max_capacity - total_demand  # 记录无人机搭载货物还未飞出时的剩余容量
        # 记录无人机在未插入前新客户前的访问每一个客户节点后的剩余容量
        Rdemand = []
        Rdemand = [0] * (route_length)
        Rdemand[0]=remaining_demand
        for i in range(1, route_length-1):
            indices = np.where(self.customers[:, 0] == route[i])[0][0]
            if self.customers[indices][3]>0:
                remaining_demand += self.customers[indices][3]
                Rdemand[i]=remaining_demand
            else:
                remaining_demand -= abs(self.customers[indices][3])
                Rdemand[i]=remaining_demand
        Rdemand[route_length-1] = remaining_demand
        # 尝试寻找一个插入点
        for i in range(1, route_length):
            is_valid_route = True  # 标记路径是否有效
            # 先检查插入后容量是否满足
            if customer_i[3]<0:
                for j in range(i-1, route_length):
                    if Rdemand[j]-abs(customer_i[3])<0:
                        is_valid_route = False
                        break
            if customer_i[3]>0:
                for j in range(0, i):
                    if Rdemand[j]-customer_i[3]<0:
                        is_valid_route = False
                        break
            if not is_valid_route:
                continue
            modified_route = route[:i] + [customer_i[0]] + route[i:]                          #更新插入后的路径
            prev_indices = np.where(self.customers[:, 0] == modified_route[i-1])[0][0]-1      #插入节点前的索引
            curent_indices = np.where(self.customers[:, 0] == modified_route[i])[0][0]-1      #插入节点的索引
            distance = self.distanceDmatrix[prev_indices+1][curent_indices+1]                 #距离
            arrival_time=self.T[prev_indices][4]+distance/self.drone_speed                    #无人机到达插入节点的时间
            if arrival_time>self.customers[curent_indices][5]:                                #若大于最晚服务时间，则该插入位存在问题
                continue
            wait_time=max(0, self.customers[curent_indices][4]-arrival_time)                #计算可能存在的等待时间
            depart_time=wait_time+arrival_time+self.service_time                            #离开当前节点的时间
            #检验后续节点是否满足时间约束
            for j in range(i+1 , len(modified_route)-1):
                prev_customer = modified_route[j - 1]                                       # 前一个客户
                curent_customer=modified_route[j]                                           # 当前客户
                prev_indices  = np.where(self.customers[:, 0]  == prev_customer)[0][0]-1    # 找到其在customers数组中的行索引，便于计算距离
                curent_indices = np.where(self.customers[:, 0] == curent_customer)[0][0]-1
                distance = self.distanceDmatrix[prev_indices+1][curent_indices+1]
                arrival_time = depart_time+distance / self.drone_speed
                # 检查到达时间是否在客户的时间窗内
                if  arrival_time > self.allcustomers[curent_customer - 1].end_time:
                        is_valid_route = False      # 标记为无效路径
                        break                       # 如果不符合，跳过当前插入点
                wait_time = max(0, self.allcustomers[curent_customer - 1].start_time-arrival_time)
                depart_time=arrival_time+wait_time+self.service_time
            if not is_valid_route:
                continue
            candidate_route.append(modified_route)
        if customer_i[3] > 0 and candidate_route:
            total_demand+=customer_i[3]
        route_energy=[]
        for i, existing_route in enumerate(candidate_route):
            copy_T = copy.deepcopy(self.T)  # 深拷贝时间矩阵
            first_indices = np.where(self.customers[:, 0] == existing_route[0])[0][0]-1
            #检查该路径能量消耗能不能尊重约束
            total_energy_needed = self.calculate_energy(self.T[first_indices][4], existing_route, total_demand)
            if total_energy_needed > self.max_battery:
                continue
            else:
                total_energy_needed=max(0, total_energy_needed-orignal_energy_needed)
            for k in range(1 , len(existing_route)):
                prev_customer = existing_route[k-1]                                   # 前一个客户
                curent_customer = existing_route[k]                                 # 当前客户
                prev_indices = np.where(self.customers[:, 0] == prev_customer)[0][0]-1   # 找到其在customers数组中的行索引，便于计算距离
                curent_indices = np.where(self.customers[:, 0] == curent_customer)[0][0]-1
                distance = self.distanceDmatrix[prev_indices+1][curent_indices+1]
                arrival_time=self.T[prev_indices][4]+distance/self.drone_speed
                wait_time = max(0, self.allcustomers[curent_customer - 1].start_time-arrival_time)
                if k<len(existing_route)-1:
                    self.T[curent_indices][1] = 0
                    self.T[curent_indices][2] = 0
                    self.T[curent_indices][3] = arrival_time
                    self.T[curent_indices][4] = arrival_time + wait_time + self.service_time
                    # 更新在回收节点时无人机的离开时间
                if k == len(existing_route) - 1:
                    self.T[curent_indices][3]=arrival_time
                    if self.T[curent_indices][3]>self.T[curent_indices][2]:             #假设 卡车需要等待无人机返回
                        self.T[curent_indices][2]=self.T[curent_indices][3]                 #更新 卡车的离开时间
                        if curent_customer in launch_node:                                  #假设 无人机返回后马上进行新的行程
                            self.T[curent_indices][4]=self.T[curent_indices][3]                   #更新 无人机离开时间
                        else:
                            self.T[curent_indices][4] = self.T[curent_indices][2]
                    elif self.T[curent_indices][3]<self.T[curent_indices][1]:           #假设 无人机需要悬浮等待卡车
                        if curent_customer in launch_node:                                  #假设 无人机返回后马上进行新的行程
                            self.T[curent_indices][4]=self.T[curent_indices][1]                   #更新 无人机离开时间
                        else:
                            self.T[curent_indices][4] = self.T[curent_indices][2]
                    else:                                                               #假设 不需要等待
                        if curent_customer in launch_node:                                  #假设 无人机返回后马上进行新的行程
                            self.T[curent_indices][4]=self.T[curent_indices][3]                   #更新 无人机离开时间
                        else:
                            self.T[curent_indices][4] = self.T[curent_indices][2]
            i_index=self.truck.Troute.index(existing_route[len(existing_route)-1])+1
            self.Update_T(i_index)
            for i in range(self.cnum - 1):
                if self.T[i][1] == 0:
                    if self.T[i][3] > self.allcustomers[self.T[i][0] - 1].end_time:
                        modify=1
                        break
                if self.T[i][3] == 0:
                    if self.T[i][1] > self.allcustomers[self.T[i][0] - 1].end_time:
                        modify = 1
                        break
                else:
                    if self.T[i][1] > self.allcustomers[self.T[i][0] - 1].end_time:
                        modify = 1
                        break
            if modify ==1:
                modify=0
                self.T=copy_T
                continue
            self.T = copy_T
            route_energy.append((existing_route, total_energy_needed))
        # 使用 min() 查找 total_energy_needed 最小的路径
        if  route_energy:
            min_route = min(route_energy, key=lambda x: x[1])
            complete_route = min_route[0]
            total_energy=min_route[1]
            total_energy=total_energy+orignal_energy_needed
            del self.truck.Troute[need_delete_index]  # 将客户从卡车路径中删除
            return total_energy, complete_route
        else:
            return orignal_energy_needed, []

    def calculate_energy(self, time, drone_route,demand): #传入参数 无人机起飞节点出发时间 无人机路径
        curent_load=0
        # 计算 无人机路径上的送货客户的需求之和
        for i in range(1, len(drone_route)-1):
            current_indices = np.where(self.customers[:, 0] == drone_route[i])[0][0]
            if self.customers[current_indices][3]>0:
                curent_load+=self.customers[current_indices][3]

        for i in range(1 , len(drone_route)):
            if i-1==0:
                prev_indices=np.where(self.customers[:, 0] == drone_route[0])[0][0]
                current_indices=np.where(self.customers[:, 0] == drone_route[i])[0][0]
                travel_time  = self.distanceDmatrix[prev_indices][current_indices]/self.drone_speed
                arrival_time = travel_time+time
                energy_neeed =(curent_load+self.drone_weight)*travel_time*self.energy_fight
                wait_time=max(0, self.customers[current_indices][4]-arrival_time)
                depart_time=arrival_time+wait_time+self.service_time
                energy_neeed += (curent_load + self.drone_weight) * wait_time * self.energy_hover
                energy_neeed += (curent_load + self.drone_weight) * self.service_time * self.energy_service
                if self.customers[current_indices][3] > 0:
                    curent_load -=self.customers[current_indices][3]
                else:
                    curent_load += abs(self.customers[current_indices][3])
            elif i == len(drone_route)-1:
                prev_indices = np.where(self.customers[:, 0] == drone_route[i-1])[0][0]
                current_indices = np.where(self.customers[:, 0] == drone_route[i])[0][0]
                travel_time = self.distanceDmatrix[prev_indices][current_indices] / self.drone_speed
                arrival_time += travel_time + depart_time
                energy_neeed += (curent_load + self.drone_weight) * travel_time * self.energy_fight
                wait_time = max(0, self.T[current_indices-1][2] - arrival_time)
                energy_neeed += (curent_load + self.drone_weight) * wait_time * self.energy_hover
            else:
                prev_indices = np.where(self.customers[:, 0] == drone_route[i-1])[0][0]
                current_indices = np.where(self.customers[:, 0] == drone_route[i])[0][0]
                travel_time = self.distanceDmatrix[prev_indices][current_indices] / self.drone_speed
                arrival_time += travel_time + depart_time
                energy_neeed += (curent_load + self.drone_weight) * travel_time * self.energy_fight
                wait_time = max(0, self.customers[current_indices-1][4]-arrival_time)
                depart_time = arrival_time + wait_time + self.service_time
                energy_neeed += (curent_load + self.drone_weight) * wait_time * self.energy_hover
                energy_neeed += (curent_load + self.drone_weight) * self.service_time * self.energy_service
                if self.customers[current_indices][3] > 0:
                    curent_load -= self.customers[current_indices][3]
                else:
                    curent_load += abs(self.customers[current_indices][3])
        return  energy_neeed