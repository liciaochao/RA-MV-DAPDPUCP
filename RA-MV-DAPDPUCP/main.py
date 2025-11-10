import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 导入matplotlib库的pyplot模块，用于绘图
import copy
import plot
import math
from Cla import Customer, Solution
from Cla import Truck
from Cla import Drone
from Cla import Problem
from Cluster import FCM
from TRUCK_Routes import TRUCKtsp
from DRONE_Routes import AddDroneRoute
from Dynamic_optimize import Dynamic_Optimization
from Dynamic_optimize import DestroyOperators
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置图表中文字体为宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决图表中负号显示问题

def read_data(file_path):
    df = pd.read_csv(file_path)
    customers = []
    for _, row in df.iterrows():
        customer = Customer(row['CUST_NO'], row['XCOORD'], row['YCOORD'], row['DEMAND'], row['ST'], row['ET'],  row['DE'])
        customers.append(customer)
    return customers

# 计算卡车行驶路径距离
def TRUCK_distance(customer_location) :
    customer_count = len(customer_location)                             # 客户数量
    dis = [[0] * customer_count for i in range(customer_count)]         # 初始化距离矩阵
    for i in range(customer_count):                                     # 对每一个城市
        for j in range(customer_count):                                 # 对每一个城市
            if i != j:                                                  # 如果不是同一个城市
                dis[i][j] = abs(customer_location[i][1] - customer_location[j][1]) + abs(customer_location[i][2] - customer_location[j][2])  # 计 算 距 离
            else:
                dis[i][j] = 0   # 同一个城市距离为 0
    return dis                  # 返回距离矩阵

# 计算无人机行驶路径距离
def Drone_distance(customer_location) :
    customer_count = len(customer_location)                             # 客户数量
    dis = [[0] * customer_count for i in range(customer_count)]         # 初始化距离矩阵
    for i in range(customer_count):                                     # 对每一个城市
        for j in range(customer_count):                                 # 对每一个城市
            if i != j:  # 如果不是同一个城市
                dis[i][j] = math.sqrt((customer_location[i][0] - customer_location[j][0]) ** 2 + (
                        customer_location[i][1] - customer_location[j][1]) ** 2)  # 计算距离
            else:
                dis[i][j] = 0  # 同一个城市距离为 0
    return dis  # 返回距离矩阵

if __name__ == "__main__":
    # 读取文件
    file_path = 'D:\\python\\mDAPDP-TW-UCP\\Solomon\\50_90_50.csv'
    customers = read_data(file_path)
    print(customers)
    depot = [0, 0, 0, 0, 0, 480, -1]
    #创建问题参数
    problem = Problem([depot[1], depot[2]], customers, 5, 200, 480, 10, 9, 650, 15)
    problem.print_customer()
    problem.print_depot()
    problem.plot_location()
    #将列表转为数组便于处理数据
    customers_array = np.array([[customer.cust_no, customer.xcoord, customer.ycoord, customer.demand,
                                 customer.start_time, customer.end_time, customer.drone_eligible] for customer in
                                customers])
    #------------------开始聚类-------------------------
    total_demand = problem.totalDdemand                                                                                                 # 计算总的配送需求
    num_clusters = max(1, int(total_demand / (problem.truck_max_load-60-problem.cluster_remand_demand)) + 1)                            #聚类数量
    FCMRes = FCM(customers_array, num_clusters, 30, problem.truck_max_load, problem.customer_list, 60, problem.cluster_remand_demand)                  # 初始化FCM聚类器
    print("最佳目标函数——1", FCMRes.best_J)
    print("最佳迭代次数", FCMRes.iter_num_COUT)
    print("最佳聚类中心")
    print(FCMRes.Clast)
    print("目标函数")
    print(FCMRes.Jlist)
    plot.plot_clus(FCMRes)
    plt.show()
    # 显示绘图窗口
    print("聚类信息：")
    for cluster in FCMRes.clusters:
        print(cluster)
    #-------------------------各个聚类里面构造TSP路径-------------------------
    TRUCK_Routes = []
    DRONE_Routes = []
    Initial_solution=Solution()
    Current_solution=Solution()
    Best_solution=Solution()
    Copy_solution=Solution()
    #计算所有客户的距离矩阵
    customers=np.insert(customers_array, 0, depot, 0)                      # 将选定的机场点加到路径的起始位置
    ALLdistanceTmatrix = TRUCK_distance(customers)
    ALLdistanceDmatrix = Drone_distance(customers)

    for cluster in FCMRes.clusters:                                                     # 遍历每个聚类
        customers = FCMRes.data[cluster.indices]                                        # 获取聚类中的城市点
        customers = np.insert(customers, 0, depot, 0)                          # 将选定的机场点加到路径的起始位置
        customers = customers.reshape(len(FCMRes.data[cluster.indices]) + 1, 7)         # 重新调整城市数组的形状
        distance_matrix = TRUCK_distance(customers)                                     # 计算城市之间的距离矩阵_卡车
        # 初始化并求解优化的TSP
        truck = Truck(cluster.cluster_id, problem.truck_max_load, problem.truck_v, [depot[1], depot[2]],
                      problem.truck_max_work_time)
        tsp_solver = TRUCKtsp(customers, distance_matrix, problem.truck_v, problem.truck_max_work_time,
                              problem.service_time, problem.wait_time_weight, truck, problem.customer_list, problem.drone_weight,
                              start_node=0)
        tsp_solver.solve()
        truck.Troute=tsp_solver.get_route()
        TRUCK_Routes.append(truck)
        print(f"第{cluster.cluster_id+1}条旅行路径: {truck.Troute}")
        print(f"第{cluster.cluster_id+1}条旅行路径总旅行距离: {tsp_solver.get_total_distance()}")
        print(f"第{cluster.cluster_id+1}条旅行路径总旅行时间: {tsp_solver.get_total_time()}")
    for truck in TRUCK_Routes:
        print(f"卡车编号: {truck.vehicle_id+1}, 路径: {truck.Troute}, 出发时间：{truck.begin_time}, 返回时间时间：{truck.end_time}")
    plot.plot_truck(TRUCK_Routes, problem.customer_list)
    for truck, cluster in zip(TRUCK_Routes, FCMRes.clusters):
        customers = FCMRes.data[cluster.indices]                                    # 获取聚类中的城市点
        print(customers)
        customers = np.insert(customers, 0, depot, 0)                      # 将选定的机场点加到路径的起始位置
        customers = customers.reshape(len(FCMRes.data[cluster.indices]) + 1, 7)     # 重新调整城市数组的形状
        distance_matrix = TRUCK_distance(customers)
        # # 初始化并求解优化的无人机路径
        distanceDmatrix = Drone_distance(customers)                                 # 计算客户之间的距离矩阵_无人机
        drone = Drone(cluster.cluster_id, problem.drone_max_load, problem.drone_v, problem.drone_max_endurance)
        ADD_Drone_Route = AddDroneRoute(customers, distance_matrix, distanceDmatrix, problem.truck_v, problem.drone_v,
                                        problem.drone_weight, problem.drone_max_load,
                                        problem.drone_max_endurance, problem.service_time, drone, truck,
                                        problem.customer_list, problem.energy_fight, problem.energy_service,
                                        problem.energy_hover)
        ADD_Drone_Route.assign_customers_to_drone()
        DRONE_Routes.append(drone)
# 更新卡车到达仓库的时间
    for truck in TRUCK_Routes:
        length = len(truck.Troute)
        last_node=truck.Troute[length-2]
        time=abs(customers_array[last_node-1][1] - depot[1]) + abs(customers_array[last_node-1][2] - depot[2])/problem.truck_v
        truck.end_time=time+problem.customer_list[last_node-1].departure_truck
        print(f"卡车编号: {truck.vehicle_id+1}, 路径: {truck.Troute}, 出发时间：{truck.begin_time}, 返回时间时间：{truck.end_time}")
    plot.plot_truck(TRUCK_Routes, problem.customer_list)
    plot.plot_drone_routes(TRUCK_Routes, DRONE_Routes, problem.customer_list)
    for drone in DRONE_Routes:
        print(f"无人机编号: {drone.vehicle_id + 1}")
        plot.plot_single_drone_routes(TRUCK_Routes, drone, problem.customer_list)
        for trip in drone.route:
            path = [int(x) for x in trip['path']]
            energy=trip['energy']
            # 除去 path 的第一项 和 最后一项进行求和
            delivery_total = sum(problem.customer_list[c - 1].demand for c in path[1:-1] if problem.customer_list[c - 1].demand > 0)
            pickup_total   = sum(problem.customer_list[c - 1].demand for c in path[1:-1] if problem.customer_list[c - 1].demand < 0)
            print(f" 路径: {path}, 总派送需求: {delivery_total}, 总取件需求: {pickup_total}, 总耗能: {energy}")

    # 开始动态规划
    DynamicOptimize = Dynamic_Optimization(TRUCK_Routes, DRONE_Routes, FCMRes.clusters, problem.customer_list,
                                           problem.down_delete, problem.up_delete, problem.truck_max_load,
                                           problem.truck_v, problem.drone_v,
                                           problem.drone_weight, problem.drone_max_load,
                                           problem.drone_max_endurance,
                                           problem.service_time, problem.energy_fight, problem.energy_service,
                                           problem.energy_hover, problem.cost_truck, problem.cost_drone,
                                           ALLdistanceTmatrix, ALLdistanceDmatrix, Initial_solution,
                                           Current_solution,
                                           Best_solution, Copy_solution)





    success = DynamicOptimize.run_dynamic_optimization()
    if success:
        print("\n动态优化完成！")
        print("显示最终结果...")

        # 显示最终路径
        print("\n最终路径配置：")
        for truck_id in range(len(DynamicOptimize.TRUCK_Routes)):
            assigned_customers = DynamicOptimize.get_vehicle_customers(truck_id)
            print(f"车辆对{truck_id}:")
            print(f"   负责客户: {sorted(assigned_customers)}")
            print(f"   卡车路径: {DynamicOptimize.TRUCK_Routes[truck_id].Troute}")
            print(f"   无人机任务: {len(DynamicOptimize.DRONE_Routes[truck_id].route)}个")
            total_cost = DynamicOptimize.cost_single_vehicle(truck_id)
            print(f"   总成本: {total_cost:.2f}")

        final_total_cost = DynamicOptimize.cost()
        print(f"\n最终总成本: {final_total_cost:.2f}")
    else:
        print("\n动态优化遇到问题")
    print("\n程序运行完成！")