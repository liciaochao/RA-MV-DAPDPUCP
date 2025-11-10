import numpy as np
import matplotlib.pyplot as plt     # 导入matplotlib库的pyplot模块，用于绘图
from typing import List, Dict
from copy import deepcopy

def place_holder(*args):
    """此函数不执行任何操作。它只是一个占位符，用于填充代码中的 “漏洞”。它唯一的目的是使代码看起来不错（从语法上讲）"""
    return args

class Truck:
    """此类表示卡车"""
    def __init__(self, vehicle_id, max_capacity, speed, location, max_work_time):
        self.vehicle_id = vehicle_id
        self.max_capacity = max_capacity
        self.speed = speed
        self.location = location
        self.max_work_time = max_work_time
        self.current_work_time = 0            #当前工作时间
        self.current_load = 0                 #当前载重
        self.current_load_delivery   = 0      #当前送货载重
        self.current_load_pickup = 0          #当前取货载重
        self.initial_load = 0                 #出发时候的载重
        self.initial_load_delivery   = 0      #出发时候的送货载重
        self.initial_load_pickup = 0          #出发时候的取货载重
        self.begin_time = 0                   #离开仓库时间
        self.end_time = 0                     #返回仓库时间
        self.Troute = []                      #卡车预服务路径

    def __repr__(self):
        return "<Truck {} at {}. max_capacity={}, speed={}, current_load={}, current_load_delivery={}, current_load_pickup={}, current_work_time={}, location={}, route={}>".format(
            self.vehicle_id,
            hex(id(self)),
            self.max_capacity,
            self.speed,
            self.current_load,
            self.current_load_delivery,
            self.current_load_pickup,
            self.initial_load,
            self.initial_load_delivery,
            self.initial_load_pickup,
            self.current_work_time,
            self.location,
            self.Troute)

class Drone:
    """此类表示无人机"""
    def __init__(self, vehicle_id,  max_capacity, speed, max_battery):
        self.vehicle_id = vehicle_id                     #无人机序号
        self.max_capacity = max_capacity                 #无人机最大容量
        self.speed = speed                               #无人机速度
        self.max_battery = max_battery                   #无人机最大续航
        self.location = 0                                #无人机位置
        self.route=[]                                    #无人机行程

    def add_trip(self, launch_node, retrieval_node, path, total_energy):
        """记录一个新的行程"""
        trip = {
            'launch_node': launch_node,                  # 起飞节点（数字）
            'retrieval_node': retrieval_node,            # 回收节点（数字）
            'path': path,                                # 行程路径（列表）
            'energy': total_energy,                      #行程消耗能量
            'current_remain_battery':self.max_battery,   #无人机当前能量
            'current_load': 0,                           # 无人机当前载重
            'current_load_delivery': 0,                  # 无人机当前送货载重
            'current_load_pickup': 0,                     # 无人机当前取件载重
            'initial_load': 0,                           # 无人机初始载重
            'initial_load_delivery': 0,                  # 无人机初始送货载重
            'initial_load_pickup': 0                     # 无人机初始取件载重
        }
        self.route.append(trip)  # 将行程添加到列表中

    def get_trip(self, trip_index):
        """获取特定行程的信息"""
        if 0 <= trip_index < len(self.route):
            return self.route[trip_index]
        else:
            return "行程索引不存在"

    def print_route(self):
        """按照给定的格式打印所有行程的路径"""
        print(f"无人机编号: {self.vehicle_id + 1}, 路径: {self.route}")

    def __repr__(self):
        return "<Drone {} at {}. max_capacity={}, max_battery={}, speed={},location={}>,route={}".format(
            self.vehicle_id,
            hex(id(self)),
            self.max_capacity,
            self.max_battery,
            self.speed,
            self.location,
            self.route)

class Customer:
    def __init__(self, cust_no, xcoord, ycoord, demand, start_time, end_time, drone_eligible):
        self.cust_no = cust_no                  # 客户编号
        self.xcoord = xcoord                    # 横坐标
        self.ycoord = ycoord                    # 纵坐标
        self.demand = demand                    # 客户的需求量
        self.start_time = start_time            # 客户最早服务时间
        self.end_time = end_time                # 客户最晚服务时间
        self.drone_eligible = drone_eligible    # 无人机是否可访问  0表示不可访问，1表示可访问

        self.cluster = None             # 聚类结果
        self.service_by = None          # 客服被哪对卡车/无人机服务
        self.arrive_truck = None        # 卡车到达时间
        self.arrive_drone = None        # 无人机到达时间
        self.departure_truck=None       # 卡车离开时间
        self.departure_drone = None     # 无人机离开时间
        self.service_begin = None       # 开始服务时间
        self.wait = None                # 服务等待时间

        self.launch=None                # 记录该客户节点是否作为了起飞节点
        self.retrieve = None            # 记录该客户节点是否作为了回收节点
        self.success = None             # 服务成功
        self.random = None              # 随机数 —— 用以判断服务成功概率
        self.possibility = 0.8         # 在家概率

        def copy(self, other_customer):
            # 复制必要的属性
            from main import problem
            self.xcoord = other_customer.xcoord
            self.ycoord = other_customer.ycoord
            self.demand = other_customer.demand
            self.drone_eligible = other_customer.drone_eligible
            self.cluster = other_customer.cluster
            self.service_by = other_customer.service_by
            self.launch = other_customer.launch
            self.retrieve = other_customer.retrieve
            self.arrive_truck = other_customer.arrive_truck
            self.arrive_drone = other_customer.arrive_drone
            self.departure_truck = other_customer.departure_truck
            self.departure_drone = other_customer.departure_drone
            # 其他属性修改为 None 或根据需求重新赋值
            self.cust_no = other_customer.cus_no+len(problem.customer_list)     # 客户编号可能需要重新定义或设为 None
            self.start_time = None                                              # 服务时间重新设置
            self.end_time = None                                                # 服务时间重新设置
            self.arrive = None                                                  # 到达时间重置为 None
            self.service_begin = None                                           # 服务开始时间重置为 None
            self.wait = None                                                    # 服务等待时间重置为 None
            self.departure = None                                               # 离开时间
            self.success = None                                                 # 服务成功标志重置为 None
            self.random = None                                                  # 随机数重置为 None
            self.possibility = None                                             # 在家概率重置为 None

    def __repr__(self):
        return (f"Customer(CUST_NO={self.cust_no}, XCOORD={self.xcoord}, YCOORD={self.ycoord}, "
                f"DEMAND={self.demand}, start_time={self.start_time}, end_time={self.end_time}, "
                f"drone_eligible={self.drone_eligible}, cluster={self.cluster}, "
                f"arrive_truck={self.arrive_truck}, arrive_drone={self.arrive_drone}, "
                f"service_begin={self.service_begin}, wait_time={self.wait}, "
                f"departure_truck={self.departure_truck}, departure_drone={self.departure_drone}, "
                f"service_by={self.service_by}, launch={self.launch}, retrieve={self.retrieve}, "
                f"success={self.success}, random={self.random}, possibility={self.possibility})")

class Problem:
    """该类表示一个配送问题，目标是从一个指定的仓库（depot）出发，向一系列客户（clients）提供配送服务。该类还可以存储一个解决方案列表（solutions），每个解决方案是该问题的一个可能解。"""
    def __init__(self, depot, customer_list, truck_v, truck_max_load, truck_max_work_time, drone_v, drone_max_load,
                 drone_max_endurance,drone_weight):
        self.depot = depot                                  #仓库地理信息
        self.customer_list = customer_list                  #客户信息
        self.truck_v = truck_v                              #卡车速度
        self.truck_max_load = truck_max_load                #卡车最大载重
        self.truck_max_work_time = truck_max_work_time      #卡车最大工作时间
        self.drone_v = drone_v                              #无人机速度
        self.drone_max_load = drone_max_load                #无人机最大载重
        self.drone_max_endurance = drone_max_endurance      #无人机最大续航
        self.drone_weight=drone_weight                      #无人机重量
        self.service_time = 10                              # 单个客户服务时间
        self.wait_time_weight = 1.0                         # 构造初始解的等待时间成本权重参数
        self.energy_fight=0.5                               # 无人机飞行时能量消耗系数
        self.energy_service=0.3                             # 无人机服务时能量消耗系数
        self.energy_hover=0.2                               # 无人机悬浮时能量消耗系数
        self.CAF=None                                       # 客户可用性文件
        self.down_delete=0.2                                # 一条路径中最少删除客户比例
        self.up_delete = 0.4                                # 一条路径中最多删除客户比例
        self.cost_truck = 0.078                              # 卡车单位行驶成本
        self.cost_drone = 0.00248                           # 无人机单位行驶成本
        self.cluster_remand_demand=0                        # 聚类时留下的多余容量
    @property
    def number_of_customer(self):
        return len(self.customer_list)

    @property                               #计算总的配送需求
    def totalDdemand(self):
        total = 0
        for customer in self.customer_list:
            if customer and customer.demand > 0:
                total += customer.demand
        return total

    @property                               #计算总的取件需求
    def totalPdemand(self):
        total = 0
        for customer in self.customer_list:
            if customer and customer.demand <0:
                total += customer.demand
        return total

    def print_depot(self):
        print("仓库的位置：", self.depot)

    def print_customer(self):
        for customer in self.customer_list:
            print(repr(customer))

    def plot_location(self):
        plt.figure(figsize=(10, 8))  # 设置宽度为10英寸，高度为8英寸
        # 创建包含客户位置的二维数组
        data = np.array([[customer.xcoord, customer.ycoord] for customer in self.customer_list])
        # 绘制客户的位置，客户用蓝色圆点（'ob'）
        plt.plot(data[:, 0], data[:, 1], 'ob', markersize=6, label='Customer')
        # 仓库用红色五角星表示，仓库位置是 depot[0] 和 depot[1]
        plt.plot(self.depot[0], self.depot[1], 'r*', markersize=12, label='Depot')
        # 设置图表标签和标题
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('仓库-客户分布图')
        # 显示图例
        plt.legend()
        # 显示图像
        plt.show()

class Solution:
    def __init__(self):
        """无参构造函数，通过方法逐步构建解决方案"""
        self.trucks: List[Dict] = []  # 存储卡车信息（含独立副本和成本）
        self.drones: List[Dict] = []  # 存储无人机信息
        self.customers: Dict[int, object] = {}  # {客户ID: 客户对象副本}
        self.total_cost: float = 0  # 总成本
        self.truck_costs: List[float] = []  # 每辆卡车的单独成本（含无人机协同成本）
        self.drone_costs: List[float] = []  # 每架无人机的单独成本

    def add_truck(self, truck_obj, cost: float = 0):
        """添加卡车副本并记录其成本
        Args:
            truck_obj: 卡车对象
            cost: 该卡车及其关联无人机的总成本
        """
        self.trucks.append({
            'obj': deepcopy(truck_obj),
            'cost': cost,
            'route': deepcopy(truck_obj.Troute)  # 保存路径快照
        })
        self.truck_costs.append(cost)
        self._update_total_cost()
        return self

    def add_drone(self, drone_obj, cost: float = 0):
        """添加无人机副本并记录其成本"""
        self.drones.append({
            'obj': deepcopy(drone_obj),
            'cost': cost,
            'trips': deepcopy(drone_obj.route)  # 保存任务快照
        })
        self.drone_costs.append(cost)
        self._update_total_cost()
        return self

    def add_customer(self, customer_id: int, customer_obj):
        """添加客户副本"""
        self.customers[customer_id] = deepcopy(customer_obj)
        return self

    def set_cost(self, cost: float):
        """手动设置总成本（通常用calculate_cost自动计算更安全）"""
        self.total_cost = cost
        return self

    def _update_total_cost(self):
        """内部方法：自动更新总成本"""
        self.total_cost = sum(self.truck_costs) + sum(self.drone_costs)

    def print_solution(self):
        """结构化输出解决方案"""
        print("=" * 40)
        print(f"【总成本】: {self.total_cost:.2f}")

        # 输出卡车信息
        print("\n【卡车配送方案】")
        for i, truck in enumerate(self.trucks, 1):
            print(f" 卡车{i}:")
            print(f"   - 路径: {truck['route']}")
            print(f"   - 成本: {truck['cost']:.2f}")

        # 输出无人机信息
        print("\n【无人机配送方案】")
        for j, drone in enumerate(self.drones, 1):
            print(f" 无人机{j}:")
            print(f"   - 任务数: {len(drone['trips'])}")
            print(f"   - 总能耗: {sum(t['energy'] for t in drone['trips']):.2f}")
            print(f"   - 成本: {drone['cost']:.2f}")

        # 输出客户服务情况
        print("\n【客户服务统计】")
        served = set()
        for truck in self.trucks:
            served.update(truck['route'][1:-1])  # 去掉仓库节点
        for drone in self.drones:
            for trip in drone['trips']:
                served.update(trip['path'][1:-1])
        print(f" 已服务客户数: {len(served)}/{len(self.customers)}")

    def to_dict(self) -> Dict:
        """将解决方案转为字典（便于JSON序列化）"""
        return {
            'total_cost': self.total_cost,
            'trucks': [{
                'id': t['obj'].vehicle_id,
                'route': t['route'],
                'cost': t['cost']
            } for t in self.trucks],
            'drones': [{
                'id': d['obj'].vehicle_id,
                'trips': d['trips'],
                'cost': d['cost']
            } for d in self.drones]
        }

    def copy(self):
        """创建解决方案的深拷贝"""
        new_solution = Solution()
        new_solution.trucks = deepcopy(self.trucks)
        new_solution.drones = deepcopy(self.drones)
        new_solution.customers = deepcopy(self.customers)
        new_solution.total_cost = self.total_cost
        new_solution.truck_costs = deepcopy(self.truck_costs)
        new_solution.drone_costs = deepcopy(self.drone_costs)
        return new_solution