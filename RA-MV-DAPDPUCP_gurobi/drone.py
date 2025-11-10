from gurobipy import *
import re
import math
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
from itertools import combinations




"""
定义数据类
"""


class Data(object):
    """
    存储算例数据的类
    """

    def __init__(self):
        self.customer_num = 0        # 客户数量
        self.node_num = 0            # 点的数量
        self.max_endurance = 180     # 无人机的飞行范围
        self.lunching_time = 0       # 发射无人机消耗的时间
        self.recover_time = 0        # 回收无人机消耗的时间
        self.cor_X = []              # 点的横坐标
        self.cor_Y = []              # 点的纵坐标
        self.demand = []             # 客户点的需求
        self.ready_time = []         # 最早开始时间
        self.due_time = []           # 最晚结束时间
        self.drone_eligible = []     #无人机可访问资格
        self.disTmatrix = [[]]       # 点间的曼哈顿距离矩阵
        self.disDmatrix = [[]]       # 点间的欧几里得距离矩阵
        self.CAF = []                # 客户可用性文件
        self.truck_customer = []     #只能由卡车服务的客户
        self.drone_customer = []     #能由无人机服务的客户
        self.pickup_customer = []    # 取件客户
        self.delivery_customer = []  # 送货客户
        self.drone_v=667             # 无人机速度  m/min  40 km/h
        self.truck_v =500            # 卡车速度   m/min   30 km/h
        self.service_time =2         #服务时间     min
        self.earliest_time=0         #最早发车时间
        self.latest_time = 450       #最晚返航时间
        self.fixcost_truck    = 200  #卡车的固定成本
        self.perdistance_truck= 0.00078       #卡车单位行程的成本
        self.perpower_drone   = 0.000104      #无人机单位能耗的成本
        self.perpenalize      = 100     #惩罚成本
        self.weight_drone     = 6       #无人机自重
        self.energy_factor    = 0.667   #能量损耗系数 Wh/(kg min)
        self.drone_capacity   = 10       #无人机载重
        self.truck_capacity   = 650     #卡车载重
    # 读取数据的函数
    def read_data(self, path_customer, path_caf, customer_num):
        """
        读取算例数据中前customer_num个顾客的数据。

        :param path: 文件路径
        :param customer_num: 顾客数量
        :return:
        """
        self.customer_num = customer_num
        self.node_num = customer_num + 2
        # 初始化列表，确保可以给每个索引赋值
        self.cor_X = [0] * self.node_num
        self.cor_Y = [0] * self.node_num
        self.demand = [0] * self.node_num
        self.ready_time = [0] * self.node_num
        self.due_time = [0] * self.node_num
        self.drone_eligible = [0] * self.node_num
        #设置仓库属性
        self.cor_X[0]=3000
        self.cor_Y[0]=4600
        self.demand[0]=0
        self.ready_time[0]=0
        self.due_time[0]=450
        self.drone_eligible[0]=0
        # 打开 CSV 文件并读取
        with open(path_customer, 'r') as f:
            lines = csv.reader(f)
            count = 0
        # 读取数据信息
            for line in lines:
                count = count + 1
                if (count >= 2 and count < 2 + customer_num):
                    self.cor_X[count - 1] = float(line[1])  # 确保索引正确，转换为浮动类型
                    self.cor_Y[count - 1] = float(line[2])  # 同上
                    self.demand[count - 1] = float(line[3])  # 同上
                    self.ready_time[count - 1] = float(line[4])  # 同上
                    self.due_time[count - 1] = float(line[5])  # 同上
                    self.drone_eligible[count - 1] = int(line[6])  # 同上，转换为整数
            # 复制一份depot
            self.cor_X[1 + customer_num] = 3000
            self.cor_Y[1 + customer_num] = 4600
            self.demand[1 + customer_num] = 0
            self.ready_time[1 + customer_num] = 0
            self.due_time[1 + customer_num] = 450
            self.drone_eligible[1 + customer_num] = 0
        # 设置集合 例如仅卡车能服务客户集合 无人机能服务集合 取件客户集合 送件客户集合
        for i in range(1, customer_num+1):
            if self.demand[i]>0:
                self.delivery_customer.append(i)
            else:
                self.pickup_customer.append(i)
                #设置无人机可运载客户包裹上限
            if abs(self.demand[i])>self.drone_capacity or self.drone_eligible[i]==0 :
                self.truck_customer.append(i)
            if abs(self.demand[i])<=self.drone_capacity and self.drone_eligible[i]==1:
                self.drone_customer.append(i)

        # 计算距离矩阵
        self.disTmatrix = np.zeros((self.node_num, self.node_num))
        for i in range(0, self.node_num):
            for j in range(0, self.node_num):
                self.disTmatrix[i][j] = abs(self.cor_X[i] - self.cor_X[j]) + abs(self.cor_Y[i] - self.cor_Y[j])
        self.disDmatrix = np.zeros((self.node_num, self.node_num))
        for i in range(0, self.node_num):
            for j in range(0, self.node_num):
                temp = (self.cor_X[i] - self.cor_X[j]) ** 2 + (self.cor_Y[i] - self.cor_Y[j]) ** 2
                self.disDmatrix[i][j] = math.sqrt(temp)
                temp = 0
        # 读入CAF文件数据
        with open(path_caf, 'r') as f:
            lines = csv.reader(f)
            count = 0
            # 初始化 self.CAF 为包含 customer_num 个空列表的二维结构
            self.CAF = [[] for _ in range(customer_num)]
            # 遍历每一行并将数据转换为数字
            for i, line in enumerate(lines):
                if count >= customer_num:
                    break
                for j in range(10):
                    self.CAF[i].append(float(line[j]))  # 转换为浮动数并追加到自定义的二维列表
                count += 1

    # 打印算例数据
    def print_data(self, customer_num):
        print("\n------- 无人机参数 --------")
        print("%-20s %4d" % ('UAV max_endurance: ', self.max_endurance))
        print("%-20s %4d" % ('UAV lunching time: ', self.lunching_time))
        print("%-20s %4d" % ('UAV recover time: ', self.recover_time))
        print("\n------- 点的信息 --------")
        print('%-10s %-10s %-10s %-8s %-6s %-10s %-6s' % ('横坐标', '纵坐标', '需求', '起始时间', '截止时间', '是否接受无人机配送', '服务时间'))
        for i in range(len(self.demand)):
            print('%-12.1f %-12.1f %-12.1f %-12.1f %-12.1f %-d %-12.1f' % (self.cor_X[i],self.cor_Y[i],
            self.demand[i], self.ready_time[i], self.due_time[i],self.drone_eligible[i], self.service_time))

        print("-------卡车距离矩阵-------\n")
        for i in range(self.node_num):
            for j in range(self.node_num):
                print("%6.2f" % (self.disTmatrix[i][j]), end=" ")
            print()
        print("-------无人机距离矩阵-------\n")
        for i in range(self.node_num):
            for j in range(self.node_num):
                print("%6.2f" % (self.disDmatrix[i][j]), end=" ")
            print()
        print("-------CAF文件-------\n")
        for i in range(customer_num):
            for j in range(10):
                print("%6.2f" % (self.CAF[i][j]), end=" ")
            print()
        print("-------无人机可服务客户-------\n")
        print(self.drone_customer)
        print("-------仅卡车可服务客户-------\n")
        print(self.truck_customer)
        print("-------取件客户-------\n")
        print(self.pickup_customer)
        print("-------配送客户-------\n")
        print(self.delivery_customer)
        print("----------数据打印完成！--------------------\n")

class Model_builder(object):
    """
    构建模型的类。
    """

    def __init__(self):
        self.model = None  # 模型
        self.big_M = 450
        self.time_M=450
        self.max_endurance=180
        self.max_drone_capacity =10
        self.max_truck_capacity =650
        # U[i]
        self.U = [[0] for i in range(data.node_num)]
        # U_drone[i]
        self.X_truck_with_drone = [([0] * data.node_num) for j in range(data.node_num)]
        # X_{ij}^{-}
        self.X_truck = [([0] * data.node_num) for j in range(data.node_num)]
        # Y_{ij}
        self.Y_drone = [([0] * data.node_num) for j in range(data.node_num)]
        # Y_i^{launch}
        self.Y_launch = [[0] for i in range(data.node_num)]
        # Y_i^{retrieve}
        self.Y_retrieve = [[0] for i in range(data.node_num)]
        # Z_i
        self.Z = [[0] for i in range(data.node_num)]
        # Z_drone_i
        self.Z_drone = [[0] for i in range(data.node_num)]
        # Z_truck_i
        self.Z_truck = [[0] for i in range(data.node_num)]
        # T_{k,i}^truck
        self.T_truck = [[0] for i in range(data.node_num)]
        # T_{d,i}^{drone}
        self.T_drone = [[0] for i in range(data.node_num)]
        # T_i^{service}
        self.T_service = [[0] for i in range(data.node_num)]
        # T_i^hover^s
        self.T_hover_s = [0.0 for i in range(data.node_num)]
        # T_i^hover^r
        self.T_hover_r = [0.0 for i in range(data.node_num)]
        # e_i
        self.e       = [[0] for i in range(data.node_num)]
        # e_{ij}^f
        self.e_fly = [([0] * data.node_num) for j in range(data.node_num)]
        # e_i^s
        self.e_service = [[0] for i in range(data.node_num)]
        # e_i^h
        self.e_hover = [[0] for i in range(data.node_num)]
        # w_{i,+}^{t,p}
        self.w_truck_pick_arrive = [[0] for i in range(data.node_num)]
        # w_{i,+}^{t,d}
        self.w_truck_delivery_arrive = [[0] for i in range(data.node_num)]
        # w_{i,-}^{t,p}
        self.w_truck_pick_left = [[0] for i in range(data.node_num)]
        # w_{i,-}^{t,d}
        self.w_truck_delivery_left = [[0] for i in range(data.node_num)]
        # w_{i,+}^{d,p}
        self.w_drone_pick_arrive = [[0] for i in range(data.node_num)]
        # w_{i,+}^{d,d}
        self.w_drone_delivery_arrive = [[0] for i in range(data.node_num)]
        # w_{i,-}^{d,p}
        self.w_drone_pick_left = [[0] for i in range(data.node_num)]
        # w_{i,-}^{d,d}
        self.w_drone_delivery_left = [[0] for i in range(data.node_num)]
        self.delta_1 = [[0] for i in range(data.node_num)]
        self.delta_2 = [[0] for i in range(data.node_num)]
        self.delta_3 = [[0] for i in range(data.node_num)]
        self.delta_exist = [[0] for i in range(data.node_num)]
        self.S = [[0] for i in range(data.node_num)]

        #求一个集合的所有子集
    def all_possible_subsets_of(self, C_drone):
        return [set(subset) for r in range(2, len(data.drone_customer) + 1) for subset in combinations(data.drone_customer, r)]

    def remove_Z_constraint_for_customer(self, cid):
        """
        从模型中移除客户 cid 的初始 Z[cid] == 1 约束。
        """
        if cid in self.Z_constraints:
            constr = self.Z_constraints[cid]
            self.model.remove(constr)
            self.model.update()  # 更新模型以生效删除
            del self.Z_constraints[cid]
        else:
            print(f"⚠️ 客户 {cid} 没有对应的 Z 约束或已被删除。")

    def build_model(self, data=None, solve_model=False):
        """
        构建模型的类。
        :param data: 算例数据
        :param solve_model: 是否求解模型
        :return:
        """
        # 创建模型对象
        self.model = Model("FSTSP")

        # 创建决策变量
        for i in range(data.node_num):
            name1 = 'Y_' + str(i) + '^launch'
            name2 = 'Y_' + str(i) + '^retrieve'
            name3 = 'Z_' + str(i)
            name4 = 'T_' + str(i) + '^truck'
            name5 = 'T_' + str(i) + '^drone'
            name6 = 'T_' + str(i) + '^service'
            name7 = 'T_' + str(i) + '^hover_'+'^s'
            name8 = 'T_' + str(i) + '^hover_' + '^r'
            name9 = 'e_' + str(i)
            name10 = 'e_' + str(i) + '^s'
            name11 = 'e_' + str(i) + '^h'
            name12 = 'w_'+'truck_'+'pick_'+'arrive_'+str(i)
            name13 = 'w_'+'truck_'+'delivery_'+'arrive_'+str(i)
            name14 = 'w_'+'truck_'+'pick_'+'left_'+str(i)
            name15 = 'w_'+'truck_'+'delivery_'+'left_'+str(i)
            name16 = 'w_'+'drone_'+'pick_'+'arrive_'+str(i)
            name17 = 'w_'+'drone_'+'delivery_'+'arrive_'+str(i)
            name18 = 'w_'+'drone_'+'pick_'+'left_'+str(i)
            name19 = 'w_'+'drone_'+'delivery_'+'left_'+str(i)
            name20 = 'U_' + str(i)
            name21 = 'Z_'+'drone_' + str(i)
            name22 = 'Z_'+'truck_' + str(i)

            self.Y_launch[i] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=name1)
            self.Y_retrieve[i] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=name2)
            self.Z[i] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=name3)
            self.T_truck[i] = self.model.addVar(lb=0, ub=self.time_M, vtype=GRB.CONTINUOUS, name=name4)
            self.T_drone[i] = self.model.addVar(lb=0, ub=self.time_M, vtype=GRB.CONTINUOUS, name=name5)
            self.T_service[i] = self.model.addVar(lb=0, ub=self.time_M, vtype=GRB.CONTINUOUS, name=name6)
            self.T_hover_s[i] = self.model.addVar(lb=0, ub=self.time_M, vtype=GRB.CONTINUOUS, name=name7)
            self.T_hover_r[i] = self.model.addVar(lb=0, ub=self.time_M, vtype=GRB.CONTINUOUS, name=name8)
            self.e[i] = self.model.addVar(lb=0, ub=self.max_endurance, vtype=GRB.CONTINUOUS, name=name9)
            self.e_service[i] = self.model.addVar(lb=0, ub=self.max_endurance, vtype=GRB.CONTINUOUS, name=name10)
            self.e_hover[i] = self.model.addVar(lb=0, ub=self.max_endurance, vtype=GRB.CONTINUOUS, name=name11)
            self.w_truck_pick_arrive[i] = self.model.addVar(lb=0, ub=self.max_truck_capacity, vtype=GRB.CONTINUOUS, name=name12)
            self.w_truck_delivery_arrive[i] = self.model.addVar(lb=0, ub=self.max_truck_capacity, vtype=GRB.CONTINUOUS, name=name13)
            self.w_truck_pick_left[i] = self.model.addVar(lb=0, ub=self.max_truck_capacity, vtype=GRB.CONTINUOUS, name=name14)
            self.w_truck_delivery_left[i] = self.model.addVar(lb=0, ub=self.max_truck_capacity, vtype=GRB.CONTINUOUS, name=name15)
            self.w_drone_pick_arrive[i] = self.model.addVar(lb=0, ub=self.max_drone_capacity, vtype=GRB.CONTINUOUS, name=name16)
            self.w_drone_delivery_arrive[i] = self.model.addVar(lb=0, ub=self.max_drone_capacity, vtype=GRB.CONTINUOUS, name=name17)
            self.w_drone_pick_left[i] = self.model.addVar(lb=0, ub=self.max_drone_capacity, vtype=GRB.CONTINUOUS, name=name18)
            self.w_drone_delivery_left[i] = self.model.addVar(lb=0, ub=self.max_drone_capacity, vtype=GRB.CONTINUOUS, name=name19)
            self.U[i] = self.model.addVar(lb=0, ub=data.node_num, vtype=GRB.CONTINUOUS, name=name20)
            self.Z_drone[i] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=name21)
            self.Z_truck[i] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=name22)

            for j in range(data.node_num):
                name23 = 'X_' + str(i) + "_" + str(j)+'^d'
                name24 = 'X_' + str(i) + "_" + str(j)
                name25 = 'Y_' + str(i) + "_" + str(j)
                name26 = 'e_' + str(i) + "_" + str(j) + '^f'
                self.X_truck_with_drone[i][j] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=name23)
                self.X_truck[i][j] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=name24)
                self.Y_drone[i][j] = self.model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=name25)
                self.e_fly[i][j] = self.model.addVar(lb=0, ub=self.max_endurance, vtype=GRB.CONTINUOUS, name=name26)

        # 创建目标函数
        objective = 0
        # 第一部分：与路径相关的目标
        for i in range(data.node_num):
            for j in range(data.node_num):
                if i!=j:
                    objective += (self.X_truck_with_drone[i][j] + self.X_truck[i][j]) * data.disTmatrix[i][j] * data.perdistance_truck
        # # 第二部分：与飞行、服务、悬停相关的目标
        for i in range(data.node_num):
            objective += ( self.e_service[i] + self.e_hover[i]) * data.perpower_drone
            for j in range(data.node_num):
                objective += self.e_fly[i][j]  * data.perpower_drone
        # 第三部分：起点到其他节点的路径目标
        objective += data.fixcost_truck
        # # 第四部分：配送失败的惩罚
        # for i in range(data.node_num):
        #     objective += self.Z[i] * data.perpenalize

        self.model.setObjective(objective, GRB.MINIMIZE)

        # 约束 1: 流平衡约束 卡车 (1)
        for j in range(1, data.node_num - 1):
            expr = LinExpr(0)
            for i in range(0, data.node_num - 1):
                if (i != j):
                    expr.addTerms(1, self.X_truck_with_drone[i][j])
            for k in range(1, data.node_num - 1):
                if (k != j):
                    expr.addTerms(1, self.X_truck[k][j])
            self.model.addConstr(expr == self.Z_truck[j], name=f"c1_{j}")
        # 约束 2: 流平衡约束 卡车 (2)
        for j in range(1, data.node_num - 1):
            expr = LinExpr(0)
            for k in range(1, data.node_num ):
                if (k != j):
                    expr.addTerms(1, self.X_truck_with_drone[j][k])
            for n in range(1, data.node_num - 1):
                if (n != j):
                    expr.addTerms(1, self.X_truck[j][n])
            self.model.addConstr(expr == self.Z_truck[j], name=f"c2_{j}")
        # 约束 3: 流平衡约束 无人机
        for j in  data.drone_customer:
            expr = LinExpr(0)
            for i in range(1, data.node_num - 1):
                if (i != j):
                    expr.addTerms(1, self.Y_drone[i][j])
            self.model.addConstr(expr == self.Z_drone[j], name=f"c3_{j}")
        # 约束 4: 流平衡约束 无人机
        for j in  data.drone_customer:
            expr = LinExpr(0)
            for i in range(1, data.node_num - 1):
                if (i != j):
                    expr.addTerms(1, self.Y_drone[j][i])
            self.model.addConstr(expr == self.Z_drone[j], name=f"c4_{j}")
        # 约束 5: 无人机不从起点起飞也不在终点回收
        expr = LinExpr(0)
        for i in range(1, data.node_num - 1):
            expr.addTerms(1, self.Y_drone[0][i])
        self.model.addConstr(expr == 0, "c5")
        # 约束 6: 无人机不从起点起飞也不在终点回收
        expr = LinExpr(0)
        for i in range(1, data.node_num - 1):
            expr.addTerms(1, self.Y_drone[i][data.node_num-1])
        self.model.addConstr(expr == 0, "c6")
        # 约束 7: 无人机不从起点起飞也不在终点回收
        expr = LinExpr(0)
        for i in range(1, data.node_num - 1):
            expr.addTerms(1, self.X_truck[0][i])
        self.model.addConstr(expr == 0, "c7")
        # 约束 8: 无人机不从起点起飞也不在终点回收
        expr = LinExpr(0)
        for i in range(1, data.node_num - 1):
            expr.addTerms(1, self.X_truck[i][data.node_num-1])
        self.model.addConstr(expr == 0, "c8")
        # 约束 9: 每个客户都必须得到服务
        for i in range(1, data.node_num - 1):
            expr = LinExpr(0)
            expr.addTerms(1, self.Z_truck[i])
            expr.addTerms(1, self.Z_drone[i])
            self.model.addConstr(expr == 1, "c9_{i}")
        # 约束 10: 卡车客户只能卡车服务
        for i in data.truck_customer:
            expr = LinExpr(0)
            expr.addTerms(1, self.Z_truck[i])
            self.model.addConstr(expr == 1, "c10_{i}")
        # 约束 11: 卡车从起点出发一次
        expr = LinExpr(0)
        for j in range(1, data.node_num-1):
            expr.addTerms(1, self.X_truck_with_drone[0][j])
        self.model.addConstr(expr == 1, "c11")
        # 约束 12: 卡车到达终点一次
        expr = LinExpr(0)
        for i in range(1, data.node_num - 1):
            expr.addTerms(1, self.X_truck_with_drone[i][data.node_num - 1])
        self.model.addConstr(expr == 1, "c12")

        # 约束 : 子环路消除约束
        for i in range(1, data.node_num):
            self.model.addConstr(self.U[i] >= 1, name=f"c13_{i}")
            self.model.addConstr(self.U[i] <= data.node_num-1, name=f"c14_{i}")
        self.model.addConstr(self.U[0] == 0, name=f"c15")
        for i in range(1, data.node_num-1):
            for j in range(1, data.node_num):
                if i != j:
                    self.model.addConstr(self.U[j] >= self.U[i] + 1 - data.node_num  * (
                                1 - self.X_truck_with_drone[i][j] - self.X_truck[i][j]), name=f"mtz_truck_{i}_{j}")

        # 约束 16:   卡车需要访问起飞节点
        for i in range(1, data.node_num-1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.Z_truck[i])
            expr2.addTerms(-1, self.Z_drone[i])
            for j in data.drone_customer:
                if (j != i):
                    expr2.addTerms(1, self.Y_drone[i][j])
            self.model.addConstr(expr1 >= expr2, name=f"c16_{i}")

        # 约束 17:   卡车需要访问回收节点
        for i in range(1, data.node_num-1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.Z_truck[i])
            expr2.addTerms(-1, self.Z_drone[i])
            for j in range(1, data.node_num-1):
                if (j != i):
                    expr2.addTerms(1, self.Y_drone[j][i])
            self.model.addConstr(expr1 >= expr2, name=f"c17_{i}")

        # 约束 18: 确定起飞节点
        for i in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.Y_launch[i])
            expr2.addTerms(-1, self.Z_drone[i])
            for j in data.drone_customer:
                if (j != i):
                    expr2.addTerms(1, self.Y_drone[i][j])
            self.model.addConstr(expr1 >= expr2, name=f"c18_{i}")

        # 约束 19: 确定起回收节点
        for i in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.Y_retrieve[i])
            expr2.addTerms(-1, self.Z_drone[i])
            for j in data.drone_customer:
                if (j != i):
                    expr2.addTerms(1, self.Y_drone[j][i])
            self.model.addConstr(expr1 >= expr2, name=f"c21_{i}")

        # 约束 20: 确定起回收节点
        for j in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.Y_retrieve[j])
            for i in range(1, data.node_num - 1):
                if (i != j):
                    expr2.addTerms(1, self.Y_drone[i][j])
            self.model.addConstr(expr1 <= expr2, name=f"c20_{i}")

        # 约束 21: 确定起回收节点
        for j in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.Y_launch[j])
            for i in range(1, data.node_num - 1):
                if (i != j):
                    expr2.addTerms(1, self.Y_drone[j][i])
            self.model.addConstr(expr1 <= expr2, name=f"c21_{i}")

        # 约束 22  起飞节点后面的不携带无人机的卡车路径为 1
        for i in range(1, data.node_num - 1):
            expr = LinExpr(0)
            expr.addTerms(1,self.Y_launch[i])
            for j in range(1, data.node_num - 1):
                if (j != i):
                    expr.addTerms(1, self.X_truck_with_drone[i][j])
            self.model.addConstr(expr <=1, name=f"22_{i}")

        # 约束 23  无人机在卡车上才能被放飞
        for i in range(1, data.node_num - 1):
            expr = LinExpr(0)
            for j in range(1, data.node_num - 1):
                if j != i:
                    expr.addTerms(1, self.X_truck_with_drone[j][i])
            expr.addTerms(1, self.Y_retrieve[i])
            self.model.addConstr(self.Y_launch[i] <= expr, name=f"c_launch_condition_{i}")

        # 约束 24 : 保证 “无人机起飞节点在卡车路径中必须出现在回收节点之前
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                # 当 Y_launch[i] = 1 且 Y_retrieve[j] = 1 时，必须有 U[i] + 1 <= U[j]。
                self.model.addConstr(self.U[i] + 1 <= self.U[j] + self.big_M * (2 - self.Y_launch[i] - self.Y_retrieve[j]),name=f"24_{i}_{j}")

        # 约束 25 禁止无人机自循环
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if (i != j):
                    expr1 = LinExpr(0)
                    expr2 = LinExpr(0)
                    expr1.addTerms(1, self.Y_drone[i][j])
                    expr2.addTerms(1, self.Y_drone[j][i])
                    self.model.addConstr(expr1 <= 1-expr2, name=f"c25_{i}_{j}")

        # 约束 26 生成并添加无人机子环消除约束
        for S in self.all_possible_subsets_of(data.drone_customer):
            expr = LinExpr()
            for i in S:
                for j in S:
                    if i != j:
                        expr.addTerms(1, self.Y_drone[i][j])  # 累加Y_drone[i][j]
            self.model.addConstr(expr <= len(S) - 1, name=f"subtour_elimination_{S}")

        # 约束 27  起飞节点后面的不携带无人机的卡车路径为 1
        for i in range(1, data.node_num - 1):
            expr = LinExpr(0)
            expr.addTerms(1,self.Y_launch[i])
            for j in range(1, data.node_num - 1):
                if (j != i):
                    expr.addTerms(1, self.X_truck_with_drone[i][j])
            self.model.addConstr(expr <=1, name=f"27_{i}")

        # 约束 28 限制卡车不携带无人机的变量的值
        for i in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            for j in range(1, data.node_num - 1):
                if (j != i):
                    expr1.addTerms(1, self.X_truck[i][j])
            expr2.addTerms(-1, self.Y_retrieve[i])
            expr2.addTerms(self.big_M, self.Y_launch[i])
            self.model.addConstr(expr1 <= 1+expr2, name=f"28_{i}")

        # 约束 29 继承约束 前面是卡车搭载无人机路径，且中间节点不为起飞节点，那么下一段卡车不搭载无人机路径为 0
        for j in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            for i in range(1, data.node_num - 1):
                if (i != j):
                    expr2.addTerms(1, self.X_truck[j][i])
            for k in range(0, data.node_num - 1):
                if (k != j):
                    expr1.addTerms(1, self.X_truck_with_drone[k][j])
            self.model.addConstr(expr2 <= 1-expr1+self.Y_launch[j]*self.big_M, name=f"29_{j}")

        # 约束 30 继承约束 前面是卡车不搭载无人机路径，且中间节点不为回收节点，那么下一段卡车搭载无人机路径为 0
        for j in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            for i in range(1, data.node_num - 1):
                if (i != j):
                    expr2.addTerms(1, self.X_truck_with_drone[j][i])
            for k in range(1, data.node_num - 1):
                if (k != j):
                    expr1.addTerms(1, self.X_truck[k][j])
            self.model.addConstr(expr2 <= 1 - expr1 + self.Y_retrieve[j]*self.big_M, name=f"30_{j}")

##########################################################################################################################################################################
####           时间约束        ####
        # 约束 31/32 确保在客户时间窗内服务客户
        for i in range(1, data.node_num - 1):
            self.model.addConstr(self.T_service[i]<=data.due_time[i], name=f"c31_{i}")
            self.model.addConstr(self.T_service[i]>=data.ready_time[i], name=f"c32_{i}")

        # 约束 33 确保服务开始时间晚于车辆到达时间 (卡车)
        for j in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.T_truck[j])
            expr2.addTerms(1, self.T_service[j])
            self.model.addConstr(expr1 <= expr2, name=f"c33_{j}")
        # 约束 34 确保服务开始时间晚于车辆到达时间 (无人机)
        for j in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.T_drone[j])
            expr2.addTerms(1, self.T_service[j])
            self.model.addConstr(expr1 <= expr2, name=f"c34_{j}")
        # 约束 35 跟踪从仓库出发后到达的第一个节点的时间
        for i in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            t_truck = data.disTmatrix[0][i]/data.truck_v
            expr1.addTerms(1, self.T_truck[0])
            expr1.addTerms(t_truck,self.X_truck_with_drone[0][i])
            expr1.addConstant(-self.big_M)
            expr1.addTerms(self.big_M, self.X_truck_with_drone[0][i])
            expr2.addTerms(1, self.T_truck[i])
            self.model.addConstr(expr1 <= expr2, name=f"c35_{i}")
        # 约束 36 跟踪从仓库出发后到达的第一个节点的时间
        for i in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            t_truck = data.disTmatrix[0][i]/data.truck_v
            expr1.addTerms(1, self.T_truck[0])
            expr1.addTerms(t_truck,self.X_truck_with_drone[0][i])
            expr1.addConstant(self.big_M)
            expr1.addTerms(-self.big_M, self.X_truck_with_drone[0][i])
            expr2.addTerms(1, self.T_truck[i])
            self.model.addConstr(expr1 >= expr2, name=f"c36_{i}")
        # 约束 37 时间流关系 （无人机）
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if i != j:
                    t_drone = data.disDmatrix[i][j] / data.drone_v
                    expr1 = LinExpr(0)
                    expr2 = LinExpr(0)
                    expr1.addTerms(1, self.T_service[i])
                    expr1.addConstant(data.service_time)
                    # expr1.addTerms(-data.service_time, self.Z[i])
                    expr1.addTerms(t_drone,self.Y_drone[i][j])
                    expr1.addConstant(-self.big_M)
                    expr1.addTerms(self.big_M, self.Y_drone[i][j])
                    expr2.addTerms(1, self.T_drone[j])
                    self.model.addConstr(expr1 <= expr2, name=f"c37_{i}_{j}")
        # 约束 38 时间流关系 （无人机）
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if i != j:
                    t_drone = data.disDmatrix[i][j] / data.drone_v
                    expr1 = LinExpr(0)
                    expr2 = LinExpr(0)
                    expr1.addTerms(1, self.T_service[i])
                    expr1.addConstant(data.service_time)
                    # expr1.addTerms(-data.service_time, self.Z[i])
                    expr1.addTerms(t_drone,self.Y_drone[i][j])
                    expr1.addConstant(self.big_M)
                    expr1.addTerms(-self.big_M, self.Y_drone[i][j])
                    expr2.addTerms(1, self.T_drone[j])
                    self.model.addConstr(expr1 >= expr2, name=f"c38_{i}_{j}")
        # 约束 39 时间流关系 （卡车）
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num-1):
                if i!=j:
                    t_truck = data.disTmatrix[i][j] / data.truck_v
                    expr1 = LinExpr(0)
                    expr2 = LinExpr(0)
                    expr1.addTerms(1, self.T_service[i])
                    expr1.addConstant(data.service_time)
                    # expr1.addTerms(-data.service_time, self.Z[i])
                    expr1.addTerms(t_truck, self.X_truck_with_drone[i][j])
                    expr1.addTerms(t_truck, self.X_truck[i][j])
                    expr1.addConstant(-self.big_M)
                    expr1.addTerms(self.big_M, self.X_truck_with_drone[i][j])
                    expr1.addTerms(self.big_M, self.X_truck[i][j])
                    expr2.addTerms(1, self.T_truck[j])
                    self.model.addConstr(expr1 <= expr2, name=f"c39_{i}_{j}")
        # # 约束 40 时间流关系 （卡车）
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num ):
                if i != j:
                    t_truck = data.disTmatrix[i][j] / data.truck_v
                    expr1 = LinExpr(0)
                    expr2 = LinExpr(0)
                    expr1.addTerms(1, self.T_service[i])
                    expr1.addConstant(data.service_time)
                    # expr1.addTerms(-data.service_time, self.Z[i])
                    expr1.addTerms(t_truck, self.X_truck_with_drone[i][j])
                    expr1.addTerms(t_truck, self.X_truck[i][j])
                    expr1.addConstant(self.big_M)
                    expr1.addTerms(-self.big_M, self.X_truck_with_drone[i][j])
                    expr1.addTerms(-self.big_M, self.X_truck[i][j])
                    expr2.addTerms(1, self.T_truck[j])
                    self.model.addConstr(expr1 >= expr2, name=f"c40_{i}_{j}")
        # 约束 无人机——卡车协同约束
        for j in range(1, data.node_num - 1):
            for i in range(0, data.node_num - 1):
                if i!=j:
                    self.model.addConstr(self.T_drone[j] - self.T_truck[j] <= self.time_M * (1 - self.X_truck_with_drone[i][j]), name=f"c_drone_1_{i}")
                    self.model.addConstr(self.T_truck[j] - self.T_drone[j] <= self.time_M * (1 - self.X_truck_with_drone[i][j]), name=f"c_drone_2_{i}")
        # 约束 41 回收节点处 卡车离开时间要大于无人机返回时间
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if i != j:
                    t_truck = data.disTmatrix[i][j] / data.truck_v
                    expr1 = LinExpr(0)
                    expr2 = LinExpr(0)
                    expr1.addTerms(1, self.T_drone[i])
                    expr2.addTerms(1, self.T_truck[j])
                    expr2.addTerms(-t_truck, self.X_truck_with_drone[i][j])
                    expr2.addTerms(-t_truck, self.X_truck[i][j])
                    expr2.addConstant(2*self.big_M)
                    expr2.addTerms(-self.big_M, self.Y_retrieve [i])
                    expr2.addTerms(-self.big_M, self.X_truck_with_drone[i][j])
                    expr2.addTerms(-self.big_M, self.X_truck[i][j])
                    self.model.addConstr(expr1 <= expr2, name=f"c41_{i}_{j}")
        # 约束 42 车辆最早出发时间不早于 A
        for i in range(1, data.node_num - 1):
            t_truck = data.disTmatrix[0][i] / data.truck_v
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addConstant(data.earliest_time)
            expr2.addTerms(1, self.T_truck[i])
            expr2.addTerms(-t_truck, self.X_truck_with_drone[0][i])
            expr2.addConstant(self.big_M)
            expr2.addTerms(-self.big_M, self.X_truck_with_drone[0][i])
            self.model.addConstr(expr1 <= expr2, name=f"c42_{i}")
        # 约束 43 车辆最晚返回时间不晚于 B
        for i in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.T_truck[data.node_num-1])
            expr2.addConstant(data.earliest_time)
            self.model.addConstr(expr1 <= expr2, name=f"c43_{i}")
################################################################################################################################################################################################################
        ####           载重约束        ####
        # 约束 44 动态更新卡车的送货载重
        for j in range(1, data.node_num):
            expr1 = LinExpr(0)
            expr1.addTerms(1, self.w_truck_delivery_arrive[j])
            for i in range(0, data.node_num-1):
                if i != j:
                    expr2 = LinExpr(0)
                    expr2.addTerms(1, self.w_truck_delivery_left[i])
                    expr2.addConstant(self.max_truck_capacity)
                    expr2.addTerms(-self.max_truck_capacity, self.X_truck[i][j])
                    expr2.addTerms(-self.max_truck_capacity, self.X_truck_with_drone[i][j])
                    self.model.addConstr(expr1 <= expr2, name=f"c44_{i}_{j}")
        # 约束 45 动态更新卡车的送货载重
        for j in range(1, data.node_num):
            expr1 = LinExpr(0)
            expr1.addTerms(1, self.w_truck_delivery_arrive[j])
            for i in range(0, data.node_num - 1):
                if i != j:
                    expr2 = LinExpr(0)
                    expr2.addTerms(1, self.w_truck_delivery_left[i])
                    expr2.addConstant(-self.max_truck_capacity)
                    expr2.addTerms(self.max_truck_capacity, self.X_truck[i][j])
                    expr2.addTerms(self.max_truck_capacity, self.X_truck_with_drone[i][j])
                    self.model.addConstr(expr1 >= expr2, name=f"c45_{i}_{j}")
        # 约束 46 动态更新卡车的取货载重
        for j in range(1, data.node_num):
            expr1 = LinExpr(0)
            expr1.addTerms(1, self.w_truck_pick_arrive[j])
            for i in range(0, data.node_num-1):
                if i != j:
                    expr2 = LinExpr(0)
                    expr2.addTerms(1, self.w_truck_pick_left[i])
                    expr2.addConstant(self.max_truck_capacity)
                    expr2.addTerms(-self.max_truck_capacity, self.X_truck[i][j])
                    expr2.addTerms(-self.max_truck_capacity, self.X_truck_with_drone[i][j])
                    self.model.addConstr(expr1 <= expr2, name=f"c46_{i}_{j}")
        # 约束 47 动态更新卡车的取货载重
        for j in range(1, data.node_num):
            expr1 = LinExpr(0)
            expr1.addTerms(1, self.w_truck_pick_arrive[j])
            for i in range(0, data.node_num-1):
                if i != j:
                    expr2 = LinExpr(0)
                    expr2.addTerms(1, self.w_truck_pick_left[i])
                    expr2.addConstant(-self.max_truck_capacity)
                    expr2.addTerms(self.max_truck_capacity, self.X_truck[i][j])
                    expr2.addTerms(self.max_truck_capacity, self.X_truck_with_drone[i][j])
                    self.model.addConstr(expr1 >= expr2, name=f"c47_{i}_{j}")

        # 约束 48 动态更新卡车的送货载重
        for i in data.delivery_customer:
            expr1 = LinExpr(0)
            expr2 = QuadExpr(0)
            expr1.addTerms(1, self.w_truck_delivery_left[i])
            expr2.addTerms(1, self.w_truck_delivery_arrive[i])
            expr2.addTerms(-data.demand[i], self.Z[i])
            expr2.add(-self.w_drone_delivery_left[i] * self.Y_launch[i])
            expr2.add(self.w_drone_delivery_arrive[i] * self.Y_retrieve[i])
            expr2.addConstant(self.max_truck_capacity)
            expr2.addTerms(-self.max_truck_capacity, self.Z_truck[i])
            self.model.addConstr(expr1 <=expr2, name=f"c48_{i}")

        # 约束 49 动态更新卡车的送货载重
        for i in data.delivery_customer:
            expr1 = LinExpr(0)
            expr2 = QuadExpr(0)
            expr1.addTerms(1, self.w_truck_delivery_left[i])
            expr2.addTerms(1, self.w_truck_delivery_arrive[i])
            expr2.addTerms(-data.demand[i], self.Z[i])
            expr2.add(-self.w_drone_delivery_left[i] * self.Y_launch[i])
            expr2.add(self.w_drone_delivery_arrive[i] * self.Y_retrieve[i])
            expr2.addConstant(-self.max_truck_capacity)
            expr2.addTerms(self.max_truck_capacity, self.Z_truck[i])
            self.model.addConstr(expr1 >= expr2, name=f"c49_{i}")

        # 约束 50 动态更新卡车的送货载重
        for i in data.pickup_customer:
            expr1 = LinExpr(0)
            expr2 = QuadExpr(0)
            expr1.addTerms(1, self.w_truck_delivery_left[i])
            expr2.addTerms(1, self.w_truck_delivery_arrive[i])
            expr2.add(-self.w_drone_delivery_left[i]*self.Y_launch[i])
            expr2.add(self.w_drone_delivery_arrive[i] * self.Y_retrieve[i])
            expr2.addConstant(self.max_truck_capacity)
            expr2.addTerms(-self.max_truck_capacity, self.Z_truck[i])
            self.model.addConstr(expr1 <=expr2, name=f"c50_{i}")

        # 约束 51 动态更新卡车的送货载重
        for i in data.pickup_customer:
            expr1 = LinExpr(0)
            expr2 = QuadExpr(0)
            expr1.addTerms(1, self.w_truck_delivery_left[i])
            expr2.addTerms(1, self.w_truck_delivery_arrive[i])
            expr2.add(-self.w_drone_delivery_left[i] * self.Y_launch[i])
            expr2.add(self.w_drone_delivery_arrive[i] * self.Y_retrieve[i])
            expr2.addConstant(- self.max_truck_capacity)
            expr2.addTerms(self.max_truck_capacity, self.Z_truck[i])
            self.model.addConstr(expr1 >= expr2, name=f"c51_{i}")

        # # 约束 52 动态更新卡车的取货载重
        for i in data.delivery_customer:
                expr1 = LinExpr(0)
                expr2 = QuadExpr(0)
                expr1.addTerms(1, self.w_truck_pick_left[i])
                expr2.addTerms(1, self.w_truck_pick_arrive[i])
                expr2.add(self.w_drone_pick_arrive[i] * self.Y_retrieve[i])
                expr2.addConstant(self.max_truck_capacity)
                expr2.addTerms(-self.max_truck_capacity, self.Z_truck[i])
                self.model.addConstr(expr1 <=expr2, name=f"c52_{i}")
        # 约束 53 动态更新卡车的取货载重
        for i in data.delivery_customer:
                expr1 = LinExpr(0)
                expr2 = QuadExpr(0)
                expr1.addTerms(1, self.w_truck_pick_left[i])
                expr2.addTerms(1, self.w_truck_pick_arrive[i])
                expr2.add(self.w_drone_pick_arrive[i] * self.Y_retrieve[i])
                expr2.addConstant(-self.max_truck_capacity)
                expr2.addTerms(self.max_truck_capacity, self.Z_truck[i])
                self.model.addConstr(expr1 >=expr2, name=f"c53_{i}")
        # 约束 54 动态更新卡车的取货载重
        for i in data.pickup_customer:
                expr1 = LinExpr(0)
                expr2 = QuadExpr(0)
                expr1.addTerms(1, self.w_truck_pick_left[i])
                expr2.addTerms(1, self.w_truck_pick_arrive[i])
                expr2.addTerms(-data.demand[i], self.Z[i])
                expr2.add(self.w_drone_pick_arrive[i] * self.Y_retrieve[i])
                expr2.addConstant(self.max_truck_capacity)
                expr2.addTerms(-self.max_truck_capacity, self.Z_truck[i])
                self.model.addConstr(expr1 <=expr2, name=f"c54_{i}")
        # 约束 55 动态更新卡车的取货载重
        for i in data.pickup_customer:
                expr1 = LinExpr(0)
                expr2 = QuadExpr(0)
                expr1.addTerms(1, self.w_truck_pick_left[i])
                expr2.addTerms(1, self.w_truck_pick_arrive[i])
                expr2.addTerms(-data.demand[i], self.Z[i])
                expr2.add(self.w_drone_pick_arrive[i] * self.Y_retrieve[i])
                expr2.addConstant(-self.max_truck_capacity)
                expr2.addTerms(self.max_truck_capacity, self.Z_truck[i])
                self.model.addConstr(expr1 >=expr2, name=f"c55_{i}")

        # 约束 56 卡车在任一节点的载重满足约束
        for i in range(0, data.node_num-1 ):
                expr1 = LinExpr(0)
                expr2 = LinExpr(0)
                expr1.addTerms(1, self.w_truck_delivery_left[i])
                expr1.addTerms(1, self.w_truck_pick_left[i])
                for j in range(1, data.node_num):
                    if j != i and not (i == 0 and j == data.node_num - 1):
                        expr1.addTerms(data.weight_drone, self.X_truck_with_drone[i][j])
                expr2.addConstant(data.truck_capacity)
                self.model.addConstr(expr1<=expr2, name=f"c56_{i}")
        # 约束 57 描述出发时 卡车的送货载重
        expr = LinExpr(0)
        for i in data.delivery_customer:
                expr.addConstant(data.demand[i])
        self.model.addConstr(self.w_truck_delivery_left[0]==expr, name=f"c57")
        # 约束  描述出发时 卡车的取货载重
        self.model.addConstr(self.w_truck_pick_left[0]==0, name=f"w_truck_pick_left_0")

        # 约束 58 在无人机起飞节点 无人机不携带取件包裹
        for i in range(1, data.node_num-1 ):
                expr1 = LinExpr(0)
                expr2 = LinExpr(0)
                expr1.addTerms(1, self.w_drone_pick_left[i])
                expr2.addConstant(data.drone_capacity)
                expr2.addTerms(-data.drone_capacity, self.Y_launch[i])
                self.model.addConstr(expr1 <=expr2, name=f"c58_{i}")
        # 约束 59 在无人机起飞节点 无人机不携带取件包裹
        for i in range(1, data.node_num-1 ):
                expr1 = LinExpr(0)
                expr2 = LinExpr(0)
                expr1.addTerms(1, self.w_drone_pick_left[i])
                expr2.addConstant(-data.drone_capacity)
                expr2.addTerms(data.drone_capacity, self.Y_launch[i])
                self.model.addConstr(expr1 >=expr2, name=f"c59_{i}")

        # 约束 60 动态更新无人机在起飞节点的送货载重
        for i in data.delivery_customer:
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_delivery_left[i])
            expr2.addTerms(1, self.w_truck_delivery_arrive[i])
            expr2.addTerms(-1, self.w_truck_delivery_left[i])
            expr2.addTerms(-data.demand[i],self.Z[i])
            expr2.addConstant( self.max_truck_capacity)
            expr2.addTerms(-self.max_truck_capacity, self.Y_launch[i])
            self.model.addConstr(expr1 <= expr2, name=f"c60_{i}")
        # 约束 61 动态更新无人机在起飞节点的送货载重
        for i in data.delivery_customer:
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_delivery_left[i])
            expr2.addTerms(1, self.w_truck_delivery_arrive[i])
            expr2.addTerms(-1, self.w_truck_delivery_left[i])
            expr2.addTerms(-data.demand[i],self.Z[i])
            expr2.addConstant( -self.max_truck_capacity)
            expr2.addTerms(self.max_truck_capacity, self.Y_launch[i])
            self.model.addConstr(expr1 >= expr2, name=f"c61_{i}")
        # 约束 62 动态更新无人机在起飞节点的送货载重
        for i in data.pickup_customer:
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_delivery_left[i])
            expr2.addTerms(1, self.w_truck_delivery_arrive[i])
            expr2.addTerms(-1, self.w_truck_delivery_left[i])
            expr2.addConstant( self.max_truck_capacity)
            expr2.addTerms(-self.max_truck_capacity, self.Y_launch[i])
            self.model.addConstr(expr1 <= expr2, name=f"c62_{i}")
        # 约束 63 动态更新无人机在起飞节点的送货载重
        for i in data.pickup_customer:
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_delivery_left[i])
            expr2.addTerms(1, self.w_truck_delivery_arrive[i])
            expr2.addTerms(-1, self.w_truck_delivery_left[i])
            expr2.addConstant( -self.max_truck_capacity)
            expr2.addTerms(self.max_truck_capacity, self.Y_launch[i])
            self.model.addConstr(expr1 >= expr2, name=f"c63_{i}")

        # 约束 68 动态更新无人机服务节点时的送货载重
        for i in set(data.drone_customer) & set(data.delivery_customer):
            # 处理同时属于两个集合的元素
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_delivery_left[i])
            expr2.addTerms(1, self.w_drone_delivery_arrive[i])
            expr2.addTerms(-data.demand[i],self.Z[i])
            expr2.addConstant( self.max_truck_capacity)
            expr2.addTerms(-self.max_truck_capacity, self.Z_drone[i])
            self.model.addConstr(expr1 <= expr2, name=f"c68_{i}")
        # 约束 69 动态更新无人机服务节点时的送货载重
        for i in set(data.drone_customer) & set(data.delivery_customer):
            # 处理同时属于两个集合的元素
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_delivery_left[i])
            expr2.addTerms(1, self.w_drone_delivery_arrive[i])
            expr2.addTerms(-data.demand[i], self.Z[i])
            expr2.addConstant(-self.max_truck_capacity)
            expr2.addTerms(self.max_truck_capacity, self.Z_drone[i])
            self.model.addConstr(expr1 >= expr2, name=f"c69_{i}")
        # 约束 70 动态更新无人机服务节点时的取货载重
        for i in set(data.drone_customer) & set(data.delivery_customer):
            # 处理同时属于两个集合的元素
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_pick_left[i])
            expr2.addTerms(1, self.w_drone_pick_arrive[i])
            expr2.addConstant( self.max_truck_capacity)
            expr2.addTerms(-self.max_truck_capacity, self.Z_drone[i])
            self.model.addConstr(expr1 <= expr2, name=f"c70_{i}")
        # 约束 71 动态更新无人机服务节点时的取货载重
        for i in set(data.drone_customer) & set(data.delivery_customer):
            # 处理同时属于两个集合的元素
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_pick_left[i])
            expr2.addTerms(1, self.w_drone_pick_arrive[i])
            expr2.addConstant(-self.max_truck_capacity)
            expr2.addTerms(self.max_truck_capacity, self.Z_drone[i])
            self.model.addConstr(expr1 >= expr2, name=f"c71_{i}")
        # 约束 72 动态更新无人机服务节点时的送货载重
        for i in set(data.drone_customer) & set(data.pickup_customer):
            # 处理同时属于两个集合的元素
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_delivery_left[i])
            expr2.addTerms(1, self.w_drone_delivery_arrive[i])
            expr2.addConstant( self.max_truck_capacity)
            expr2.addTerms(-self.max_truck_capacity, self.Z_drone[i])
            self.model.addConstr(expr1 <= expr2, name=f"c72_{i}")
        # 约束 73 动态更新无人机服务节点时的送货载重
        for i in set(data.drone_customer) & set(data.pickup_customer):
            # 处理同时属于两个集合的元素
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_delivery_left[i])
            expr2.addTerms(1, self.w_drone_delivery_arrive[i])
            expr2.addConstant(-self.max_truck_capacity)
            expr2.addTerms(self.max_truck_capacity, self.Z_drone[i])
            self.model.addConstr(expr1 >= expr2, name=f"c73_{i}")
        # 约束 74 动态更新无人机服务节点时的取货载重
        for i in set(data.drone_customer) & set(data.pickup_customer):
            # 处理同时属于两个集合的元素
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_pick_left[i])
            expr2.addTerms(1, self.w_drone_pick_arrive[i])
            expr2.addTerms(-data.demand[i],self.Z_drone[i])
            expr2.addConstant(self.max_truck_capacity)
            expr2.addTerms(-self.max_truck_capacity, self.Z_drone[i])
            self.model.addConstr(expr1 <= expr2, name=f"c74_{i}")
        # 约束 75 动态更新无人机服务节点时的取货载重
        for i in set(data.drone_customer) & set(data.pickup_customer):
            # 处理同时属于两个集合的元素
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_pick_left[i])
            expr2.addTerms(1, self.w_drone_pick_arrive[i])
            expr2.addTerms(-data.demand[i],self.Z_drone[i])
            expr2.addConstant(-self.max_truck_capacity)
            expr2.addTerms(self.max_truck_capacity, self.Z_drone[i])
            self.model.addConstr(expr1 >= expr2, name=f"c75_{i}")
        # # 约束 76 77 动态更新无人机的送货载重的继承
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if i == j:
                    continue
                self.model.addConstr(
                    self.w_drone_delivery_arrive[j] - self.w_drone_delivery_left[i] <=
                    self.max_truck_capacity * (1 - self.Y_drone[i][j]),
                    name=f"c76_{i}_{j}"
                )
                self.model.addConstr(
                    self.w_drone_delivery_arrive[j] - self.w_drone_delivery_left[i] >=
                    -self.max_truck_capacity * (1 - self.Y_drone[i][j]),
                    name=f"c77_{i}_{j}"
                )
        # 约束 78 动态更新无人机的取货载重的继承
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if i == j:
                    continue
                # 只在 Y[i][j] = 1 时限制 w_pick_arrive[j] = w_pick_left[i]
                self.model.addConstr(
                    self.w_drone_pick_arrive[j] - self.w_drone_pick_left[i] <=
                    self.max_truck_capacity * (1 - self.Y_drone[i][j]),
                    name=f"c78_{i}_{j}"
                )
                self.model.addConstr(
                    self.w_drone_pick_arrive[j] - self.w_drone_pick_left[i] >=
                    -self.max_truck_capacity * (1 - self.Y_drone[i][j]),
                    name=f"c79_{i}_{j}"
                )

        # 约束 80 无人机任一节点满足载重
        for i in data.drone_customer:
            expr1 = LinExpr(0)
            expr1.addTerms(1, self.w_drone_delivery_left[i])
            expr1.addTerms(1, self.w_drone_pick_left[i])
            self.model.addConstr(expr1 <= self.max_drone_capacity, name=f"c80_{i}")

##########################################################################################################################################################################
####           载重约束        ####
        # 约束 81 量化无人机在弧上飞行的能量
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if i != j:
                    expr = LinExpr(0)
                    expr.addTerms(1, self.e_fly[i][j])
                    self.model.addConstr(expr >= 0, name=f"c81_{i}_{j}")
        # 约束 82 量化无人机在弧上飞行的能量
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if j != i:
                    expr1 = LinExpr(0)
                    expr2 = LinExpr(0)
                    expr1.addTerms(1, self.e_fly[i][j])
                    expr2.addTerms(self.max_endurance, self.Y_drone[i][j])
                    self.model.addConstr(expr1 <= expr2, name=f"c82_{i}_{j}")
        # 约束 83 量化无人机在弧上飞行的能量
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if j != i:
                    t_drone = data.disDmatrix[i][j]/ data.drone_v
                    expr1 = LinExpr(0)
                    expr2 = LinExpr(0)
                    expr1.addConstant(data.energy_factor * data.weight_drone*t_drone)
                    expr1.addTerms(data.energy_factor*t_drone, self.w_drone_delivery_left[i])
                    expr1.addTerms(data.energy_factor*t_drone, self.w_drone_pick_left[i])
                    expr1.addConstant(-data.max_endurance)
                    expr1.addTerms(data.max_endurance, self.Y_drone[i][j])
                    expr2.addTerms(1, self.e_fly[i][j])
                    self.model.addConstr(expr1 <= expr2, name=f"c83_{i}_{j}")
        # 约束 84 量化无人机在弧上飞行的能量
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if i != j:
                    t_drone = data.disDmatrix[i][j] / data.drone_v
                    expr1 = LinExpr(0)
                    expr2 = LinExpr(0)
                    expr2.addConstant(data.energy_factor * data.weight_drone * t_drone)
                    expr2.addTerms(data.energy_factor * t_drone, self.w_drone_delivery_left[i])
                    expr2.addTerms(data.energy_factor * t_drone, self.w_drone_pick_left[i])
                    expr2.addConstant(data.max_endurance)
                    expr2.addTerms(-data.max_endurance, self.Y_drone[i][j])
                    expr1.addTerms(1, self.e_fly[i][j])
                    self.model.addConstr(expr1 <= expr2, name=f"c84_{i}_{j}")
        # 约束 85 量化无人机在服务客户点  i  时的能量
        for i in range(1, data.node_num - 1):
                expr = LinExpr(0)
                expr.addTerms(1, self.e_service[i])
                expr.addTerms(1, self.e_hover[i])
                self.model.addConstr(expr >= 0, name=f"c85_{i}")
        # 约束 86 量化无人机在服务客户点 i 时的能量
        for i in data.drone_customer:
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.e_service[i])
            expr1.addTerms(1, self.e_hover[i])
            expr2.addTerms(data.max_endurance, self.Z_drone[i])
            self.model.addConstr(expr1 <= expr2, name=f"c86_{i}")
        # 约束 87 量化无人机在服务客户点 i 时的能量
        for i in data.drone_customer:
            expr1 = QuadExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(data.energy_factor * data.weight_drone * data.service_time, self.Z[i])
            expr1.add(data.energy_factor * data.service_time * self.w_drone_delivery_arrive[i] * self.Z[i])
            expr1.add(data.energy_factor * data.service_time * self.w_drone_pick_arrive[i] * self.Z[i])
            expr1.addTerms(data.energy_factor * data.weight_drone, self.T_hover_s[i])
            expr1.add(data.energy_factor*self.w_drone_delivery_arrive[i] * self.T_hover_s[i])
            expr1.add(data.energy_factor*self.w_drone_pick_arrive[i] * self.T_hover_s[i])
            expr1.addConstant(-data.max_endurance)
            expr1.addTerms(data.max_endurance, self.Z_drone[i])
            expr2.addTerms(1, self.e_service[i])
            expr2.addTerms(1, self.e_hover[i])
            self.model.addConstr(expr1 <= expr2, name=f"c87_{i}")
        # 约束 88 量化无人机在服务客户点i时的能量
        for i in data.drone_customer:
            expr1 = QuadExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(data.energy_factor * data.weight_drone * data.service_time, self.Z[i])
            expr1.add(data.energy_factor * data.service_time * self.w_drone_delivery_arrive[i] * self.Z[i])
            expr1.add(data.energy_factor * data.service_time * self.w_drone_pick_arrive[i] * self.Z[i])
            expr1.addTerms(data.energy_factor * data.weight_drone, self.T_hover_s[i])
            expr1.add(data.energy_factor * self.w_drone_delivery_arrive[i] * self.T_hover_s[i])
            expr1.add(data.energy_factor * self.w_drone_pick_arrive[i] * self.T_hover_s[i])
            expr1.addConstant(data.max_endurance)
            expr1.addTerms(-data.max_endurance, self.Z_drone[i])
            expr2.addTerms(1, self.e_service[i])
            expr2.addTerms(1, self.e_hover[i])
            self.model.addConstr(expr1 >= expr2, name=f"c88_{i}")
        # 约束 89 量化无人机在客户点 i 回收 时的能量
        for i in range(1, data.node_num - 1):
            expr = LinExpr(0)
            expr.addTerms(1, self.e_hover[i])
            self.model.addConstr(expr >= 0, name=f"c89_{i}")
        # 约束 90 量化无人机在客户点 i 回收 时的能量
        for i in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.e_hover[i])
            expr2.addTerms(self.max_endurance, self.Y_retrieve[i])
            self.model.addConstr(expr1 <= expr2, name=f"c90_{i}")
        # 约束 91 量化无人机在客户点 i 回收悬挂时消耗的能量
        for i in range(1, data.node_num - 1):
            expr1 = QuadExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(data.energy_factor * data.weight_drone, self.T_hover_r[i])
            expr1.add(data.energy_factor * self.w_drone_delivery_arrive[i] * self.T_hover_r[i])
            expr1.add(data.energy_factor * self.w_drone_pick_arrive[i] * self.T_hover_r[i])
            expr1.addConstant(-data.max_endurance)
            expr1.addTerms(data.max_endurance, self.Y_retrieve[i])
            expr2.addTerms(1, self.e_hover[i])
            self.model.addConstr(expr1 <= expr2, name=f"c91_{i}")
        # 约束 92 量化无人机在客户点 i 回收悬挂时消耗的能量
        for i in range(1, data.node_num - 1):
            expr1 = QuadExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(data.energy_factor * data.weight_drone, self.T_hover_r[i])
            expr1.add(data.energy_factor * self.w_drone_delivery_arrive[i] * self.T_hover_r[i])
            expr1.add(data.energy_factor * self.w_drone_pick_arrive[i] * self.T_hover_r[i])
            expr1.addConstant(data.max_endurance)
            expr1.addTerms(-data.max_endurance, self.Y_retrieve[i])
            expr2.addTerms(1, self.e_hover[i])
            self.model.addConstr(expr1 >= expr2, name=f"c92_{i}")
        # # 约束  线性化悬浮时间
        # for i in range(1, data.node_num - 1):
        #     expr = LinExpr()
        #     for j in range(1, data.node_num - 1):
        #         expr.addTerms(1, self.Y_drone[j][i])
        #         # ❶ 只有在无人机访问 i（expr > 0）时，才施加 delta 约束
        #     self.model.addConstr(expr <= self.time_M * self.delta_exist[i], name=f"c_check_{i}")
        #     # 确保 delta 互斥
        #     self.model.addConstr(self.delta_1[i] + self.delta_2[i] + self.delta_3[i] == self.delta_exist[i], name=f"c33-1_{i}")
        #     # delta 变量与 tau_retrieve, z_i 的关系
        #     self.model.addConstr(self.delta_1[i] <= 1 - self.Y_launch[i], name=f"c33-2_{i}")
        #     self.model.addConstr(self.delta_1[i] <= 1 - self.Z[i], name=f"c33-3_{i}")
        #     self.model.addConstr(self.delta_2[i] <= 1 - self.Y_retrieve[i], name=f"c33-4_{i}")  # 修正变量
        #     self.model.addConstr(self.delta_2[i] <= self.Z[i], name=f"c33-5_{i}")
        #      # 强化 delta_3 和 Y_retrieve[i] 的关系
        #     self.model.addConstr(self.delta_3[i] == self.Y_retrieve[i], name=f"c33-6_{i}")  # 修正为不等式
        #     # 第一种情况 (self.Y_retrieve[i] = 0, self.Z[i] = 0)   服务客户成功
        #     self.model.addConstr(
        #         self.T_hover[i] - (self.T_service[i] - self.T_drone[i]) <= self.time_M * (1 - self.delta_1[i]),
        #         name=f"c33-7_{i}")
        #     self.model.addConstr(
        #     self.T_hover[i] - (self.T_service[i] - self.T_drone[i]) >= -self.time_M * (1 - self.delta_1[i]),
        #             name=f"c33-8_{i}")
        #     # 第二种情况 (self.Y_retrieve[i] = 0, self.Z[i] = 1) 服务客户失败
        #     self.model.addConstr(self.T_hover[i] >= 0, name=f"c33-9_{i}")
        #     self.model.addConstr(self.T_hover[i] >= data.ready_time[i] - self.T_drone[i], name=f"c33-10_{i}")
        #     self.model.addConstr(self.T_hover[i] <= self.time_M * self.delta_2[i], name=f"c33-11_{i}")
        #     # 第三种情况 (self.Y_retrieve[i] = 1)
        #     self.model.addConstr(self.T_hover[i] >= self.T_truck[i] - self.T_drone[i], name=f"c33-12_{i}")  # 修正索引
        #     self.model.addConstr(self.T_hover[i] <= self.time_M * self.delta_3[i], name=f"c33-13_{i}")  # 修正索引

        # 约束 93 无人机起飞时能量为满
        for i in range(1, data.node_num - 1):
            expr = LinExpr(0)
            expr.addTerms(1, self.e[i])
            self.model.addConstr(expr >= 0, name=f"c93_{i}")
        # 约束 94 无人机起飞时能量为满
        for i in range(1, data.node_num - 1):
            expr1 = LinExpr(0)
            expr2 = LinExpr(0)
            expr1.addTerms(1, self.e[i])
            expr2.addConstant(self.max_endurance)
            expr2.addTerms(-self.max_endurance, self.Y_launch[i])
            self.model.addConstr(expr1 <= expr2, name=f"c94_{i}")
        # 约束 95 跟踪无人机的损耗能量
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if j!=i:
                    expr1 = LinExpr(0)
                    expr2 = LinExpr(0)
                    expr1.addTerms(1, self.e[i])
                    expr1.addTerms(1, self.e_fly[i][j])
                    expr1.addTerms(1, self.e_service[j])
                    expr1.addTerms(1, self.e_hover[j])
                    expr1.addConstant(-self.max_endurance)
                    expr1.addTerms(self.max_endurance, self.Y_drone[i][j])
                    # expr1.addTerms(self.max_endurance, self.Z_drone[j])
                    expr2.addTerms(1, self.e[j])
                    self.model.addConstr(expr1 <= expr2, name=f"c95_{i}_{j}")
        # 约束 96 跟踪无人机的损耗能量
        for i in range(1, data.node_num - 1):
            for j in range(1, data.node_num - 1):
                if j!=i:
                    expr1 = LinExpr(0)
                    expr2 = LinExpr(0)
                    expr1.addTerms(1, self.e[i])
                    expr1.addTerms(1, self.e_fly[i][j])
                    expr1.addTerms(1, self.e_service[j])
                    expr1.addTerms(1, self.e_hover[j])
                    expr1.addConstant(self.max_endurance)
                    expr1.addTerms(-self.max_endurance, self.Y_drone[i][j])
                    # expr1.addTerms(-self.max_endurance, self.Z_drone[j])
                    expr2.addTerms(1, self.e[j])
                    self.model.addConstr(expr1 >= expr2, name=f"c96_{i}_{j}")

        # 约束 97 无人机顺利飞回回收节点
        for i in range(1, data.node_num - 1):
            expr = LinExpr(0)
            expr.addTerms(1, self.e[i])
            for j in data.drone_customer:
                if j!=i:
                    expr.addTerms(1, self.e_fly[i][j])
                    expr.addTerms(1, self.e_hover[j])
                    expr.addConstant(-2*self.max_endurance)
                    expr.addTerms(self.max_endurance, self.Y_drone[i][j])
                    expr.addTerms(self.max_endurance, self.Y_retrieve[j])
                    self.model.addConstr(expr <= self.max_endurance, name=f"c97_{i}_{j}")
        # 约束 98 避免陷入自循环
        for i in range(1, data.node_num - 1):
                expr = LinExpr(0)
                expr.addTerms(1, self.X_truck_with_drone[i][i])
                self.model.addConstr(expr<=0, name=f"c98_{i}")
        # 约束 99 避免陷入自循环
        for i in range(1, data.node_num - 1):
                expr = LinExpr(0)
                expr.addTerms(1, self.X_truck[i][i])
                self.model.addConstr(expr<=0, name=f"c99_{i}")
        # 约束 100 避免陷入自循环
        for i in range(1, data.node_num - 1):
                expr = LinExpr(0)
                expr.addTerms(1, self.Y_drone[i][i])
                self.model.addConstr(expr<=0, name=f"c100_{i}")
        # 约束 101 设置卡车出发的初始时间
        self.model.addConstr(self.T_truck[0]==0, name=f"c101")
        # 约束 102 设置无人机出发的初始时间
        self.model.addConstr(self.T_drone[0]==0, name=f"c102")


        self.Z_constraints = {}  # 用于记录约束对象
        for i in range(1, data.node_num - 1):
            constr = self.model.addConstr(self.Z[i] == 1, name=f"Z_{i}_initial")
            self.Z_constraints[i] = constr

        for i in range(1, data.node_num - 1):
                expr1 = LinExpr(0)
                expr2 = LinExpr(0)
                expr1.addTerms(1, self.Y_launch[i])
                expr2.addTerms(1, self.Y_retrieve[i])
        self.model.addConstr(expr1==expr2, name=f"c102")

        # 求解模型

        self.model.write('DRONE.lp')
        self.model.setParam('TimeLimit', 180)
        self.model.setParam('Presolve', -1)                 #设置预处理策略。-1 表示让 Gurobi 自动选择是否使用预处理
        self.model.setParam('MIPFocus', 2)         #设置 MIP（混合整数规划）求解的关注点，2 表示“专注于改善下界”
        self.model.setParam('Cuts', 3)             #设置切割平面（cutting planes）的使用强度
        self.model.setParam('ImproveStartTime', 0) #设置启发式搜索（Heuristic）开始改进的时间点
        self.model.setParam('Heuristics', 0.5)     #设置启发式方法的强度，0.5 表示启发式方法的“中等”强度。
        # self.model.setParam('MIPGap', 0.0001)        #设置目标 gap（即上界和下界之间的差距）
        self.model.setParam('NonConvex', 2)        #设置模型是否包含非凸优化问题。2 表示启用非凸求解
        self.model.setParam('NodefileStart', 0.5)  #设置节点文件的保存策略

        if solve_model:
            self.model.optimize()
            # 检查模型状态
            if self.model.status == GRB.OPTIMAL:
                print("模型求解成功！")
            elif self.model.status == GRB.INFEASIBLE:
                print("模型不可行，正在计算IIS...")
                self.model.computeIIS()
                self.model.write("infeasible_model.ilp")
            else:
                print(f"模型状态: {self.model.status}")
        else:
            print("跳过优化，直接计算IIS...")
            self.model.computeIIS()
            self.model.write("infeasible_model.ilp")

class Solution(object):
    """
    Solution类，用于记录关于解的信息。
    """
    def __init__(self):
        self.ObjVal = 0  # 目标函数值
        # X_{ij}
        self.X_truck_with_drone = [[]]
        # X_{ij}^{-}
        self.X_truck = [[]]
        # Y_{ij}
        self.Y_drone = [[]]
        # Y_i^{launch}
        self.Y_launch = []
        # Y_i^{retrieve}
        self.Y_retrieve = []
        # Z_i
        self.Z = []
        # Z_truck_i
        self.Z_truck = []
        # Z_drone_i
        self.Z_drone = []
        # T_{k,i}^truck
        self.T_truck = []
        # T_{d,i}^{drone}
        self.T_drone = []
        # T_i^{service}
        self.T_service = []
        # T_i^hover_s
        self.hover_s = []
        # T_i^hover_r
        self.hover_r = []
        # e_i
        self.e = []
        # e_{ij}^f
        self.e_fly = [[]]
        # e_i^s
        self.e_service = []
        # e_i^h
        self.e_hover = []
        # w_{i,+}^{t,p}
        self.w_truck_pick_arrive = []
        # w_{i,+}^{t,d}
        self.w_truck_delivery_arrive = []
        # w_{i,-}^{t,p}
        self.w_truck_pick_left = []
        # w_{i,-}^{t,d}
        self.w_truck_delivery_left = []
        # w_{i,+}^{d,p}
        self.w_drone_pick_arrive = []
        # w_{i,+}^{d,d}
        self.w_drone_delivery_arrive = []
        # w_{i,-}^{d,p}
        self.w_drone_pick_left = []
        # w_{i,-}^{d,d}
        self.w_drone_delivery_left = []
        self.delta_1= []
        self.delta_2= []
        self.delta_3 = []
        self.delta_exist = []
        self.U  =[]

        self.route_truck = []  # 卡车的路径
        self.route_UAV = []    # 无人机的路径

    def get_solution(self, data, model):
        """
        从模型获得解的信息。
        :param data: 算例数据对象
        :param model: 模型
        :return:
        """
        # 检查模型状态
        # if model.status == GRB.OPTIMAL:
        self.ObjVal = model.ObjVal  # 获取目标函数值
        # X_{ij}
        self.X_truck_with_drone = [([0] * data.node_num) for j in range(data.node_num)]
        # X_{ij}^{-}
        self.X_truck = [([0] * data.node_num) for j in range(data.node_num)]
        # Y_{ij}
        self.Y_drone = [([0] * data.node_num) for j in range(data.node_num)]
        # Y_i^{launch}
        self.Y_launch = [[0] for i in range(data.node_num)]
        # Y_i^{retrieve}
        self.Y_retrieve = [[0] for i in range(data.node_num)]
        # Z_i
        self.Z = [[0] for i in range(data.node_num)]
        # Z_truck_i
        self.Z_truck = [[0] for i in range(data.node_num)]
        # Z_drone_i
        self.Z_drone = [[0] for i in range(data.node_num)]
        # T_{k,i}^truck
        self.T_truck = [[0] for i in range(data.node_num)]
        # T_{d,i}^{drone}
        self.T_drone = [[0] for i in range(data.node_num)]
        # T_i^{service}
        self.T_service = [[0] for i in range(data.node_num)]
        # T_i^hover_s
        self.T_hover_s = [0.0 for i in range(data.node_num)]
        # T_i^hover_s
        self.T_hover_r = [0.0 for i in range(data.node_num)]
        # e_i
        self.e       = [[0] for i in range(data.node_num)]
        # e_{ij}^f
        self.e_fly = [([0] * data.node_num) for j in range(data.node_num)]
        # e_i^s
        self.e_service = [[0] for i in range(data.node_num)]
        # e_i^h
        self.e_hover = [[0] for i in range(data.node_num)]
        # w_{i,+}^{t,p}
        self.w_truck_pick_arrive = [[0] for i in range(data.node_num)]
        # w_{i,+}^{t,d}
        self.w_truck_delivery_arrive = [[0] for i in range(data.node_num)]
        # w_{i,-}^{t,p}
        self.w_truck_pick_left = [[0] for i in range(data.node_num)]
        # w_{i,-}^{t,d}
        self.w_truck_delivery_left = [[0] for i in range(data.node_num)]
        # w_{i,+}^{d,p}
        self.w_drone_pick_arrive = [[0] for i in range(data.node_num)]
        # w_{i,+}^{d,d}
        self.w_drone_delivery_arrive = [[0] for i in range(data.node_num)]
        # w_{i,-}^{d,p}
        self.w_drone_pick_left = [[0] for i in range(data.node_num)]
        # w_{i,-}^{d,d}
        self.w_drone_delivery_left = [[0] for i in range(data.node_num)]
        self.delta_1 = [[0] for i in range(data.node_num)]
        self.delta_2 = [[0] for i in range(data.node_num)]
        self.delta_3 = [[0] for i in range(data.node_num)]
        self.delta_exist = [[0] for i in range(data.node_num)]
        self.U = [[0] for i in range(data.node_num)]

        # 遍历模型中的所有变量
        for m in model.getVars():
            # 使用正则表达式按 "_" 或 "^" 分割变量名称
            str_split = re.split(r"_|\^", m.VarName)
            # 根据变量名称的第一个部分判断变量类型
            if str_split[0] == "Y" and len(str_split) == 3:
                # 处理 Y_launch 和 Y_retrieve
                if "launch" in str_split:
                    i = int(str_split[1])
                    if m.X > 0.5:
                        self.Y_launch[i] = round(m.X, 0)
                    else:
                        self.Y_launch[i] = 0
                elif "retrieve" in str_split:
                    i = int(str_split[1])
                    if m.X > 0.5:
                        self.Y_retrieve[i] = round(m.X, 0)
                    else:
                        self.Y_retrieve[i] = 0
                elif str_split[2].isdigit():
                    i = int(str_split[1])
                    j = int(str_split[2])
                    if m.X > 0.5:
                        self.Y_drone[i][j] = round(m.X, 0)
                    else:
                        self.Y_drone[i][j] = 0
            # 处理 Z_i
            elif str_split[0] == "Z" :
                if len(str_split) == 2:
                    i = int(str_split[1])
                    if m.X > 0.5:
                        self.Z[i] = round(m.X, 0)
                    else:
                        self.Z[i] = 0
                else:
                    if str_split[1] == "truck":
                        if m.X > 0.5:
                            self.Z_truck[i] = round(m.X, 0)
                        else:
                            self.Z_truck[i] = 0
                    else:
                        if m.X > 0.5:
                            self.Z_drone[i] = round(m.X, 0)
                        else:
                            self.Z_drone[i] = 0
            # 处理 T_truck, T_drone, T_service
            elif str_split[0] == "T" and len(str_split) == 3:
                i = int(str_split[1])
                if str_split[2] == "truck":
                    if m.X > 0:
                        self.T_truck[i] = m.X
                    else:
                        self.T_truck[i] = 0
                elif str_split[2] == "drone":
                    if m.X > 0:
                        self.T_drone[i] = m.X
                    else:
                        self.T_drone[i] = 0
                elif str_split[2] == "service":
                    if m.X > 0:
                        self.T_service[i] = m.X
                    else:
                        self.T_service[i] = 0
            # 处理  T_hover
            elif str_split[0] == "T" and len(str_split) == 4:
                i = int(str_split[3])
                if str_split[3] == "s":
                    if m.X > 0:
                        self.T_hover_s[i] = m.X
                    else:
                        self.T_hover_s[i] = 0
                elif str_split[3] == "r":
                    if m.X > 0:
                        self.T_hover_r[i] = m.X
                    else:
                        self.T_hover_r[i] = 0
            # 处理 e_i, e_i^s, e_i^h
            elif str_split[0] == "e":
                if len(str_split) == 2:
                    i = int(str_split[1])
                    if m.X > 0:
                        self.e[i] = m.X
                    else:
                        self.e[i] = 0
                elif len(str_split) == 3:
                    i = int(str_split[1])
                    if str_split[2] == "s":
                        if m.X > 0:
                            self.e_service[i] = m.X
                        else:
                            self.e_service[i] = 0
                    elif str_split[2] == "h" :
                        if m.X > 0:
                            self.e_hover[i] = m.X
                        else:
                            self.e_hover[i] = 0
            # 处理 w_truck_pick_arrive_i 等变量
            elif str_split[0] == "w" :
                vehicle_type = str_split[1]
                action = str_split[2]
                direction = str_split[3]
                i = int(str_split[4])  # 修正索引错误
                if m.X > 0:
                    if vehicle_type == "truck":
                        if action == "pick" and direction == "arrive":
                            self.w_truck_pick_arrive[i] = m.X
                        elif action == "delivery" and direction == "arrive":
                            self.w_truck_delivery_arrive[i] = m.X
                        elif action == "pick" and direction == "left":
                            self.w_truck_pick_left[i] = m.X
                        elif action == "delivery" and direction == "left":
                            self.w_truck_delivery_left[i] = m.X
                    elif vehicle_type == "drone":
                        if action == "pick" and direction == "arrive":
                            self.w_drone_pick_arrive[i] = m.X
                        elif action == "delivery" and direction == "arrive":
                            self.w_drone_delivery_arrive[i] = m.X
                        elif action == "pick" and direction == "left":
                            self.w_drone_pick_left[i] = m.X
                        elif action == "delivery" and direction == "left":
                            self.w_drone_delivery_left[i] = m.X
                else:
                    if vehicle_type == "truck":
                        if action == "pick" and direction == "arrive":
                            self.w_truck_pick_arrive[i] = 0
                        elif action == "delivery" and direction == "arrive":
                            self.w_truck_delivery_arrive[i] = 0
                        elif action == "pick" and direction == "left":
                            self.w_truck_pick_left[i] = 0
                        elif action == "delivery" and direction == "left":
                            self.w_truck_delivery_left[i] = 0
                    elif vehicle_type == "drone":
                        if action == "pick" and direction == "arrive":
                            self.w_drone_pick_arrive[i] = 0
                        elif action == "delivery" and direction == "arrive":
                            self.w_drone_delivery_arrive[i] = 0
                        elif action == "pick" and direction == "left":
                            self.w_drone_pick_left[i] = 0
                        elif action == "delivery" and direction == "left":
                            self.w_drone_delivery_left[i] = 0
            # 处理 X_truck_with_drone[i][j]
            elif str_split[0] == "X" and len(str_split) == 4:
                i = int(str_split[1])
                j = int(str_split[2])
                if m.X > 0.5:
                    self.X_truck_with_drone[i][j] = round(m.X, 0)
                else:
                    self.X_truck_with_drone[i][j] = 0
            # 处理 X_truck[i][j]
            elif str_split[0] == "X" and len(str_split) == 3:
                i = int(str_split[1])
                j = int(str_split[2])
                if m.X > 0.5:
                    self.X_truck[i][j] = round(m.X, 0)
                else:
                    self.X_truck[i][j] = 0
            # 处理 e_fly[i][j]
            elif str_split[0] == "e" and len(str_split) == 4 and str_split[3] == "f":
                i = int(str_split[1])
                j = int(str_split[2])
                if m.X > 0:
                    self.e_fly[i][j] = m.X
                else:
                    self.e_fly[i][j]=0
            # 处理 U[i]
            elif str_split[0] == "U":
                i = int(str_split[1])
                if m.X > 0:
                    self.U[i] = m.X
                else:
                    self.U[i] = 0

        with open("solution_variables.txt", "w") as f:
            for v in model.getVars():
                f.write(f"{v.varName} = {v.X}\n")
        print("变量值已保存到 solution_variables.txt")
        print("输出卡车变量值")
        for i in range(data.node_num):
            for j in range(data.node_num):
                if i != j:
                    if self.X_truck_with_drone[i][j] > 0.5:
                        print(f"卡车_无人机路径: {i} -> {j}, 变量值: {self.X_truck_with_drone[i][j]}")
                    if self.X_truck[i][j] > 0.5:
                        print(f"卡车路径: {i} -> {j}, 变量值: {self.X_truck[i][j]}")
        print("输出结束卡车变量值")
        # 遍历所有可能的无人机路径
        print("输出无人机变量值")
        for i in range(data.node_num):
            for j in range(data.node_num):
                if i != j:
                    if self.Y_drone[i][j] > 0.5:  # 0.5 是因为 Y_drone 是 0-1 变量
                        print(f"无人机路径: {i} -> {j}, 变量值: {self.Y_drone[i][j]}")
        print("输出结束无人机变量值")
        print("输出客户服务归属")
        for i in range(data.node_num):
            if self.Z_drone[i] > 0.5:
                print(f"客户 {i} 由无人机服务")
            if self.Z_truck[i] > 0.5:
                print(f"客户 {i} 由卡车服务")
        print("输出结束客户服务归属")
        # 求解后打印 Y_launch 和 Y_retrieve
        print("无人机起飞点:")
        for i in range(data.node_num):
            if self.Y_launch[i] > 0.5:
                print(f"客户 {i} 可以作为起飞点")
        print("无人机回收点:")
        for j in range(data.node_num):
            if self.Y_retrieve[j] > 0.5:
                print(f"客户 {j} 可以作为回收点")

        # 提取truck和无人机的路径
        print('提取truck的路径')
        current_node = 0
        self.route_truck.append(current_node)
        while (current_node != data.node_num - 1):
            # print('提取truck的路径')
            for j in range(data.node_num):
                if (self.X_truck_with_drone[current_node][j] == 1 or self.X_truck[current_node][j] == 1):
                    if (j != data.node_num - 1):
                        self.route_truck.append(j)
                    current_node = j
                    break
        self.route_truck.append(0)
        print('提取truck的路径：结束')
        # 提取出无人机的路径
        """
        提取所有无人机的飞行路径，支持节点既是起飞点又是回收点
        :return: 所有飞行路径的列表
        """
        # 找到所有起飞点
        launch_nodes = [i for i in range(1, data.node_num - 1) if self.Y_launch[i] == 1]
        # 找到所有回收点
        retrieve_nodes = [i for i in range(1, data.node_num - 1) if self.Y_retrieve[i] == 1]
        # 遍历每个起飞点，提取路径
        for start_node in launch_nodes:
            temp = []  # 存储当前飞行路径
            current_node = start_node
            temp.append(current_node)  # 将起点添加到路径中
            # 提取从 start_node 到某个回收点的路径
            while True:
                next_node = None
                # 找到下一个节点
                for j in range(1, data.node_num - 1):
                    if j == current_node:
                        continue
                    if self.Y_drone[current_node][j] == 1:
                        next_node = j
                        break
                # 如果没有找到下一个节点，抛出异常
                if next_node is None:
                    raise ValueError(f"无法找到从节点 {current_node} 到回收点的路径")
                # 将下一个节点添加到路径中
                temp.append(next_node)
                current_node = next_node
                # 如果当前节点是回收点，结束当前飞行的路径提取
                if current_node in retrieve_nodes:
                    self.route_UAV.append(temp)
                    break
        print("\n\n 最优目标值：%.2f " % self.ObjVal)
        print("\n\n ------ 卡车的路径 ------- ")
        print('[', end='')
        for i in range(len(self.route_truck)):
            print(" %d " % self.route_truck[i], end=" ")
        print(']')
        print("\n\n ------ 无人机的路径 ------- ")
        for route in self.route_UAV:
            print('[', end='')
            for i in range(0, len(route)):
                print(" %d " % route[i], end=" ")
            print(']')

    def plot_solution(self, file_name=None, customer_num=0):
        """
        将解进行可视化。

        :param file_name: 存储的文件名
        :param customer_num: 顾客的数量
        :return:
        """

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        font_dict_1 = {'family': 'Arial',  # serif
                       'style': 'normal',  # 'italic',
                       'weight': 'normal',
                       'color': 'darkred',
                       'size': 30,}

        font_dict_2 = {'family': 'Arial',  # serif
                       'style': 'normal',  # 'italQic',
                       'weight': 'normal',
                       'size': 24,}

        # ax.set_aspect('equal')
        # ax.grid(which='minor', axis='both')
        # ax.set_xlim(0, 100)
        # ax.set_ylim(0, 100)

        ax.set_xlabel(r'$x$', fontdict=font_dict_1)
        ax.set_ylabel(r'$y$', fontdict=font_dict_1)
        # ax.set_xticklabels(labels=[38, 39, 40, 41, 42, 43, 44, 45, 46, 47], fontdict=font_dict_2, minor=False)
        # ax.set_yticklabels(labels=[50, 52.5, 55, 57.5, 60, 65, 70, 75], fontdict=font_dict_2, minor=False)

        title_text_str = 'Optimal Solution for FSTSP (' + str(customer_num) + ' customers)'
        ax.set_title(title_text_str, fontdict=font_dict_1)

        # 画出depot和顾客
        # ax.scatter(data.cor_X[0], data.cor_Y[0], c='black', alpha=1, marker='s', s=20, linewidths=5, label='depot')
        # ax.scatter(data.cor_X[1:-1], data.cor_Y[1:-1], c='gray', alpha=1, marker='p', s=15, linewidths=5, label='customer')  # c='red'定义为红色，alpha是透明度，marker是画的样式
        line_depot_x = [data.cor_X[0]]
        line_depot_y = [data.cor_Y[0]]

        line_customer_x = []
        line_customer_y = []
        for i in range(1, len(data.cor_X) - 1):
            line_customer_x.append(data.cor_X[i])
            line_customer_y.append(data.cor_Y[i])

        line_1, = ax.plot(line_depot_x, line_depot_y, c='black', marker='s', markersize=20, linewidth=0, zorder=10)
        line_2, = ax.plot(line_customer_x, line_customer_y, c='brown', marker='p', markersize=15, linewidth=0, zorder=10)

        # 画出卡车的路线
        line_data_truck_x = []
        line_data_truck_y = []
        line_text_x_coor = []
        line_text_y_coor = []
        for i in range(len(self.route_truck) - 1):
            current_node = self.route_truck[i]
            line_data_truck_x.append(data.cor_X[current_node])
            line_data_truck_y.append(data.cor_Y[current_node])

            line_text_x_coor.append(data.cor_X[current_node] - 0.2)
            line_text_y_coor.append(data.cor_Y[current_node])
            plt.text(data.cor_X[current_node] - 0.2, data.cor_Y[current_node], str(current_node), fontdict=font_dict_2)
        line_data_truck_x.append(data.cor_X[self.route_truck[-1]])
        line_data_truck_y.append(data.cor_Y[self.route_truck[-1]])

        # 画出无人机的路线
        line_data_UAV_x = []
        line_data_UAV_y = []
        line_UAV_text_x_coor = []
        line_UAV_text_y_coor = []

        for uav_route in self.route_UAV:
            temp_x = []
            temp_y = []

            for node in uav_route:
                temp_x.append(data.cor_X[node])
                temp_y.append(data.cor_Y[node])

            # 记录路径坐标
            if len(temp_x) > 1:  # 确保路径有效
                line_data_UAV_x.append(temp_x)
                line_data_UAV_y.append(temp_y)

            # 记录文本标注（仅在中间路径节点上标注）
            for node in uav_route[1:-1]:  # 排除起点和终点
                line_UAV_text_x_coor.append(data.cor_X[node] - 0.2)
                line_UAV_text_y_coor.append(data.cor_Y[node])
                plt.text(data.cor_X[node] - 0.2, data.cor_Y[node], str(node), fontdict=font_dict_2)

        line_3, = ax.plot(line_data_truck_x, line_data_truck_y, c='blue', linewidth=3)
        line_4, = ax.plot(line_data_UAV_x[0], line_data_UAV_y[0], c='red', linewidth=3, linestyle=':')
        for i in range(1, len(line_data_UAV_x)):
            ax.plot(line_data_UAV_x[i], line_data_UAV_y[i], c='red', linewidth=3, linestyle=':')

        # 画出箭头
        for i in range(len(self.route_truck) - 1):
            start = self.route_truck[i]
            end = self.route_truck[i + 1]
            plt.arrow(data.cor_X[start], data.cor_Y[start], data.cor_X[end] - data.cor_X[start],
                      data.cor_Y[end] - data.cor_Y[start],
                      length_includes_head=True, head_width=0.1, head_length=0.5, lw=3,
                      color='blue')

        ax.legend([line_1, line_2, line_3, line_4],
                  ['depot', 'customer', 'truck route', 'UAV route'],
                  loc='best', prop=font_dict_2
                  # , bbox_to_anchor=(1, 1, 0, 0)  # 分别控制比例，x, y, width, height
                  )

        # 导出图片
        plt.savefig(file_name)
        plt.show()


class RollingController(object):
    def __init__(self, data_obj, model_builder_obj, solution_obj,customerNum):
        """
        初始化滚动控制器：
        使用已求解的模型和解，设置滚动顺序
        """
        self.data = data_obj
        self.model_build = model_builder_obj
        self.model = model_builder_obj.model
        self.solution = solution_obj
        self.customerNum = customerNum
        self.ObjVal = 0  # 目标函数值
        # 客户访问顺序：按开始服务时间排序
        self.sequence = sorted(
            range(1, self.customerNum + 1),
            key=lambda cid: self.solution.T_service[cid]
        )
        self.frozen_customers = []
        self.failed_customers = []
        # 在家概率矩阵
        self.CAF = self.data.CAF
        self.horizon = 600
        self.num_intervals = len(self.CAF[0])
        self.current_time = 0
        self.simulated = {}

        # 定义存储变量值
        n = data.node_num
        self.X_truck_with_drone = [[0] * n for _ in range(n)]
        self.X_truck = [[0] * n for _ in range(n)]
        self.Y_drone = [[0] * n for _ in range(n)]
        self.Y_launch = [0] * n
        self.Y_retrieve = [0] * n
        self.Z = [0] * n
        self.Z_truck = [0] * n
        self.Z_drone = [0] * n
        self.T_truck = [0.0] * n
        self.T_drone = [0.0] * n
        self.T_service = [0.0] * n
        self.T_hover_s = [0.0] * n
        self.T_hover_r = [0.0] * n
        self.e = [0.0] * n
        self.e_fly = [[0.0] * n for _ in range(n)]
        self.e_service = [0.0] * n
        self.e_hover = [0.0] * n
        self.w_truck_pick_arrive = [0.0] * n
        self.w_truck_delivery_arrive = [0.0] * n
        self.w_truck_pick_left = [0.0] * n
        self.w_truck_delivery_left = [0.0] * n
        self.w_drone_pick_arrive = [0.0] * n
        self.w_drone_delivery_arrive = [0.0] * n
        self.w_drone_pick_left = [0.0] * n
        self.w_drone_delivery_left = [0.0] * n
        self.U = [0.0] * n



    def simulate_availability(self, cid):
        T_serv = self.solution.T_service[cid]
        interval_length = self.horizon / self.num_intervals
        idx = min(int(T_serv // interval_length), self.num_intervals - 1)
        prob = self.CAF[cid-1][idx]
        num=random.random()
        return num <= prob
    def freeze_variable(self, var_name, value):
        var = self.model.getVarByName(var_name)
        if var is not None:
            val = float(value)
            var.setAttr("LB", val)
            var.setAttr("UB", val)
            self.model.update()  # 很关键，确保模型内部同步
            print(f"[Freeze] {var.VarName}: LB={var.LB}, UB={var.UB}, Set to {val}")
    def freeze_prefix(self):
        # 冻结已访问客户前缀及相关决策变量
        for idx, cid in enumerate(self.frozen_customers):
            if idx!=len(self.frozen_customers)-1:
                # 服务成功与否标志
                z_val = 1 if self.simulated[cid] else 0
                self.freeze_variable(f"Z_{cid}", z_val)
                # 服务载体指示

                self.freeze_variable(f"Z_truck_{cid}", self.Z_truck[cid])
                self.freeze_variable(f"Z_drone_{cid}", self.Z_drone[cid])
                # prev = self.frozen_customers[idx-1] if idx > 0 else 0
                # # 卡车路径和无人机路径
                # self.freeze_variable(f"X_{prev}_{cid}", self.solution.X_truck[prev][cid])
                # self.freeze_variable(f"X_{prev}_{cid}^d", self.solution.X_truck_with_drone[prev][cid])
                # self.freeze_variable(f"Y_{prev}_{cid}", self.solution.Y_drone[prev][cid])
                # 起飞/回收节点
                # self.freeze_variable(f"Y_{cid}^launch", self.solution.Y_launch[cid])
                # self.freeze_variable(f"Y_{cid}^retrieve", self.solution.Y_retrieve[cid])
                # 时间变量
                self.freeze_variable(f"T_{cid}^truck", self.T_truck[cid])
                self.freeze_variable(f"T_{cid}^drone", self.T_drone[cid])
                self.freeze_variable(f"T_{cid}^service", self.T_service[cid])
                self.freeze_variable(f"T_{cid}^hover_^s", self.T_hover_s[cid])
                self.freeze_variable(f"T_{cid}^hover_^r", self.T_hover_r[cid])
                # 能耗变量
                self.freeze_variable(f"e_{cid}", self.e[cid])
                self.freeze_variable(f"e_{cid}^s", self.e_service[cid])
                self.freeze_variable(f"e_{cid}^h", self.e_hover[cid])
                # self.freeze_variable(f"e_{prev}_{cid}^f", self.solution.e_fly[prev][cid])
                # # 载重变量
                # self.freeze_variable(f"w_truck_pick_arrive_{cid}", self.solution.w_truck_pick_arrive[cid])
                # self.freeze_variable(f"w_truck_delivery_arrive_{cid}", self.solution.w_truck_delivery_arrive[cid])
                # self.freeze_variable(f"w_drone_pick_arrive_{cid}", self.solution.w_drone_pick_arrive[cid])
                # self.freeze_variable(f"w_drone_delivery_arrive_{cid}", self.solution.w_drone_delivery_arrive[cid])
                # self.freeze_variable(f"w_truck_pick_left_{cid}", self.solution.w_truck_pick_left[cid])
                # self.freeze_variable(f"w_truck_delivery_left_{cid}", self.solution.w_truck_delivery_left[cid])
                # self.freeze_variable(f"w_drone_pick_left_{cid}", self.solution.w_drone_pick_left[cid])
                # self.freeze_variable(f"w_drone_delivery_left_{cid}", self.solution.w_drone_delivery_left[cid])
                for prev in self.frozen_customers[:idx]:
                    self.freeze_variable(f"X_{prev}_{cid}", self.X_truck[prev][cid])
                    self.freeze_variable(f"X_{prev}_{cid}^d", self.X_truck_with_drone[prev][cid])
                    self.freeze_variable(f"Y_{prev}_{cid}", self.Y_drone[prev][cid])
                    # self.freeze_variable(f"e_{prev}_{cid}^f", self.solution.e_fly[prev][cid])
            else:
                    # 服务成功与否标志
                    z_val = 1 if self.simulated[cid] else 0
                    self.freeze_variable(f"Z_{cid}", z_val)
                    # 服务载体指示
                    self.freeze_variable(f"Z_truck_{cid}", self.Z_truck[cid])
                    self.freeze_variable(f"Z_drone_{cid}", self.Z_drone[cid])
                    # prev = self.frozen_customers[idx - 1] if idx > 0 else 0
                    # 卡车路径和无人机路径
                    # self.freeze_variable(f"X_{prev}_{cid}", self.solution.X_truck[prev][cid])
                    # self.freeze_variable(f"X_{prev}_{cid}^d", self.solution.X_truck_with_drone[prev][cid])
                    # self.freeze_variable(f"Y_{prev}_{cid}", self.solution.Y_drone[prev][cid])
                    # 起飞/回收节点
                    self.freeze_variable(f"Y_{cid}^launch", self.Y_launch[cid])
                    self.freeze_variable(f"Y_{cid}^retrieve", self.Y_retrieve[cid])
                    # 时间变量
                    self.freeze_variable(f"T_{cid}^truck", self.T_truck[cid])
                    self.freeze_variable(f"T_{cid}^drone", self.T_drone[cid])
                    self.freeze_variable(f"T_{cid}^service", self.T_service[cid])
                    self.freeze_variable(f"T_{cid}^hover_^s", self.T_hover_s[cid])
                    self.freeze_variable(f"T_{cid}^hover_^r", self.T_hover_r[cid])
                    # 能耗变量
                    self.freeze_variable(f"e_{cid}^h", self.e_hover[cid])
                    # self.freeze_variable(f"e_{prev}_{cid}^f", self.solution.e_fly[prev][cid])

                    # 载重变量
                    # self.freeze_variable(f"w_truck_pick_arrive_{cid}", self.solution.w_truck_pick_arrive[cid])
                    # self.freeze_variable(f"w_truck_delivery_arrive_{cid}", self.solution.w_truck_delivery_arrive[cid])
                    # self.freeze_variable(f"w_drone_pick_arrive_{cid}", self.solution.w_drone_pick_arrive[cid])
                    # self.freeze_variable(f"w_drone_delivery_arrive_{cid}", self.solution.w_drone_delivery_arrive[cid])
                    for prev in self.frozen_customers[:idx]:
                        self.freeze_variable(f"X_{prev}_{cid}", self.X_truck[prev][cid])
                        self.freeze_variable(f"X_{prev}_{cid}^d", self.X_truck_with_drone[prev][cid])
                        self.freeze_variable(f"Y_{prev}_{cid}", self.Y_drone[prev][cid])
                        # self.freeze_variable(f"e_{prev}_{cid}^f", self.solution.e_fly[prev][cid])

    def copy_from_solution(self):
        self.Z_truck = list(self.solution.Z_truck)
        self.Z_drone = list(self.solution.Z_drone)
        self.X_truck = [list(row) for row in self.solution.X_truck]
        self.X_truck_with_drone = [list(row) for row in self.solution.X_truck_with_drone]
        self.Y_drone = [list(row) for row in self.solution.Y_drone]
        self.Y_launch = list(self.solution.Y_launch)
        self.Y_retrieve = list(self.solution.Y_retrieve)
        self.T_truck = list(self.solution.T_truck)
        self.T_drone = list(self.solution.T_drone)
        self.T_service = list(self.solution.T_service)
        self.T_hover_s = list(self.solution.T_hover_s)
        self.T_hover_r = list(self.solution.T_hover_r)
        self.e = list(self.solution.e)
        self.e_service = list(self.solution.e_service)
        self.e_hover = list(self.solution.e_hover)
        self.e_fly = [list(row) for row in self.solution.e_fly]
        self.w_truck_pick_arrive = list(self.solution.w_truck_pick_arrive)
        self.w_truck_delivery_arrive = list(self.solution.w_truck_delivery_arrive)
        self.w_drone_pick_arrive = list(self.solution.w_drone_pick_arrive)
        self.w_drone_delivery_arrive = list(self.solution.w_drone_delivery_arrive)
        self.w_truck_pick_left = list(self.solution.w_truck_pick_left)
        self.w_truck_delivery_left = list(self.solution.w_truck_delivery_left)
        self.w_drone_pick_left = list(self.solution.w_drone_pick_left)
        self.w_drone_delivery_left = list(self.solution.w_drone_delivery_left)

    def reset(self, data=None):
        """
        清空所有解存储，包括路径列表
        """
        if data is None:
            return
        n = data.node_num
        self.X_truck_with_drone = [[0] * n for _ in range(n)]
        self.X_truck = [[0] * n for _ in range(n)]
        self.Y_drone = [[0] * n for _ in range(n)]
        self.Y_launch = [0] * n
        self.Y_retrieve = [0] * n
        self.Z = [0] * n
        self.Z_truck = [0] * n
        self.Z_drone = [0] * n
        self.T_truck = [0.0] * n
        self.T_drone = [0.0] * n
        self.T_service = [0.0] * n
        self.T_hover_s = [0.0] * n
        self.T_hover_r = [0.0] * n
        self.e = [0.0] * n
        self.e_fly = [[0.0] * n for _ in range(n)]
        self.e_service = [0.0] * n
        self.e_hover = [0.0] * n
        self.w_truck_pick_arrive = [0.0] * n
        self.w_truck_delivery_arrive = [0.0] * n
        self.w_truck_pick_left = [0.0] * n
        self.w_truck_delivery_left = [0.0] * n
        self.w_drone_pick_arrive = [0.0] * n
        self.w_drone_delivery_arrive = [0.0] * n
        self.w_drone_pick_left = [0.0] * n
        self.w_drone_delivery_left = [0.0] * n
        self.U = [0.0] * n
        self.route_truck = []  # 清空卡车路径列表
        self.route_UAV = []  # 清空无人机路径列表
    def extract_solution(self, model, data):
        """
        从 Gurobi 模型中提取变量值，清空旧数据后重新构建
        """
        # 清空旧解，包括路径列表
        self.reset(data)
        # 获取目标函数值
        self.ObjVal = model.ObjVal
        # 遍历模型中的所有变量
        for m in model.getVars():
            # 使用正则表达式按 "_" 或 "^" 分割变量名称
            str_split = re.split(r"_|\^", m.VarName)
            # 根据变量名称的第一个部分判断变量类型
            if str_split[0] == "Y" and len(str_split) == 3:
                # 处理 Y_launch 和 Y_retrieve
                if "launch" in str_split:
                    i = int(str_split[1])
                    if m.X > 0.5:
                        self.Y_launch[i] = round(m.X, 0)
                    else:
                        self.Y_launch[i] = 0
                elif "retrieve" in str_split:
                    i = int(str_split[1])
                    if m.X > 0.5:
                        self.Y_retrieve[i] = round(m.X, 0)
                    else:
                        self.Y_retrieve[i] = 0
                elif str_split[2].isdigit():
                    i = int(str_split[1])
                    j = int(str_split[2])
                    if m.X > 0.5:
                        self.Y_drone[i][j] = round(m.X, 0)
                    else:
                        self.Y_drone[i][j] = 0
            # 处理 Z_i
            elif str_split[0] == "Z":
                if len(str_split) == 2:
                    i = int(str_split[1])
                    if m.X > 0.5:
                        self.Z[i] = round(m.X, 0)
                    else:
                        self.Z[i] = 0
                else:
                    if str_split[1] == "truck":
                        if m.X > 0.5:
                            self.Z_truck[i] = round(m.X, 0)
                        else:
                            self.Z_truck[i] = 0
                    else:
                        if m.X > 0.5:
                            self.Z_drone[i] = round(m.X, 0)
                        else:
                            self.Z_drone[i] = 0
            # 处理 T_truck, T_drone, T_service
            elif str_split[0] == "T" and len(str_split) == 3:
                i = int(str_split[1])
                if str_split[2] == "truck":
                    if m.X > 0:
                        self.T_truck[i] = m.X
                    else:
                        self.T_truck[i] = 0
                elif str_split[2] == "drone":
                    if m.X > 0:
                        self.T_drone[i] = m.X
                    else:
                        self.T_drone[i] = 0
                elif str_split[2] == "service":
                    if m.X > 0:
                        self.T_service[i] = m.X
                    else:
                        self.T_service[i] = 0
            # 处理  T_hover
            elif str_split[0] == "T" and len(str_split) == 4:
                i = int(str_split[3])
                if str_split[3] == "s":
                    if m.X > 0:
                        self.T_hover_s[i] = m.X
                    else:
                        self.T_hover_s[i] = 0
                elif str_split[3] == "r":
                    if m.X > 0:
                        self.T_hover_r[i] = m.X
                    else:
                        self.T_hover_r[i] = 0
            # 处理 e_i, e_i^s, e_i^h
            elif str_split[0] == "e":
                if len(str_split) == 2:
                    i = int(str_split[1])
                    if m.X > 0:
                        self.e[i] = m.X
                    else:
                        self.e[i] = 0
                elif len(str_split) == 3:
                    i = int(str_split[1])
                    if str_split[2] == "s":
                        if m.X > 0:
                            self.e_service[i] = m.X
                        else:
                            self.e_service[i] = 0
                    elif str_split[2] == "h":
                        if m.X > 0:
                            self.e_hover[i] = m.X
                        else:
                            self.e_hover[i] = 0
            # 处理 w_truck_pick_arrive_i 等变量
            elif str_split[0] == "w":
                vehicle_type = str_split[1]
                action = str_split[2]
                direction = str_split[3]
                i = int(str_split[4])  # 修正索引错误
                if m.X > 0:
                    if vehicle_type == "truck":
                        if action == "pick" and direction == "arrive":
                            self.w_truck_pick_arrive[i] = m.X
                        elif action == "delivery" and direction == "arrive":
                            self.w_truck_delivery_arrive[i] = m.X
                        elif action == "pick" and direction == "left":
                            self.w_truck_pick_left[i] = m.X
                        elif action == "delivery" and direction == "left":
                            self.w_truck_delivery_left[i] = m.X
                    elif vehicle_type == "drone":
                        if action == "pick" and direction == "arrive":
                            self.w_drone_pick_arrive[i] = m.X
                        elif action == "delivery" and direction == "arrive":
                            self.w_drone_delivery_arrive[i] = m.X
                        elif action == "pick" and direction == "left":
                            self.w_drone_pick_left[i] = m.X
                        elif action == "delivery" and direction == "left":
                            self.w_drone_delivery_left[i] = m.X
                else:
                    if vehicle_type == "truck":
                        if action == "pick" and direction == "arrive":
                            self.w_truck_pick_arrive[i] = 0
                        elif action == "delivery" and direction == "arrive":
                            self.w_truck_delivery_arrive[i] = 0
                        elif action == "pick" and direction == "left":
                            self.w_truck_pick_left[i] = 0
                        elif action == "delivery" and direction == "left":
                            self.w_truck_delivery_left[i] = 0
                    elif vehicle_type == "drone":
                        if action == "pick" and direction == "arrive":
                            self.w_drone_pick_arrive[i] = 0
                        elif action == "delivery" and direction == "arrive":
                            self.w_drone_delivery_arrive[i] = 0
                        elif action == "pick" and direction == "left":
                            self.w_drone_pick_left[i] = 0
                        elif action == "delivery" and direction == "left":
                            self.w_drone_delivery_left[i] = 0
            # 处理 X_truck_with_drone[i][j]
            elif str_split[0] == "X" and len(str_split) == 4:
                i = int(str_split[1])
                j = int(str_split[2])
                if m.X > 0.5:
                    self.X_truck_with_drone[i][j] = round(m.X, 0)
                else:
                    self.X_truck_with_drone[i][j] = 0
            # 处理 X_truck[i][j]
            elif str_split[0] == "X" and len(str_split) == 3:
                i = int(str_split[1])
                j = int(str_split[2])
                if m.X > 0.5:
                    self.X_truck[i][j] = round(m.X, 0)
                else:
                    self.X_truck[i][j] = 0
            # 处理 e_fly[i][j]
            elif str_split[0] == "e" and len(str_split) == 4 and str_split[3] == "f":
                i = int(str_split[1])
                j = int(str_split[2])
                if m.X > 0:
                    self.e_fly[i][j] = m.X
                else:
                    self.e_fly[i][j] = 0
            # 处理 U[i]
            elif str_split[0] == "U":
                i = int(str_split[1])
                if m.X > 0:
                    self.U[i] = m.X
                else:
                    self.U[i] = 0

        with open("solution_variables.txt", "w") as f:
            for v in model.getVars():
                f.write(f"{v.varName} = {v.X}\n")

        # 提取truck和无人机的路径
        print('提取truck的路径')
        current_node = 0
        self.route_truck.append(current_node)
        while (current_node != data.node_num - 1):
            # print('提取truck的路径')
            for j in range(data.node_num):
                if (self.X_truck_with_drone[current_node][j] == 1 or self.X_truck[current_node][j] == 1):
                    if (j != data.node_num - 1):
                        self.route_truck.append(j)
                    current_node = j
                    break
        self.route_truck.append(0)
        print('提取truck的路径：结束')
        # 提取出无人机的路径
        """
        提取所有无人机的飞行路径，支持节点既是起飞点又是回收点
        :return: 所有飞行路径的列表
        """
        # 找到所有起飞点
        launch_nodes = [i for i in range(1, data.node_num - 1) if self.Y_launch[i] == 1]
        # 找到所有回收点
        retrieve_nodes = [i for i in range(1, data.node_num - 1) if self.Y_retrieve[i] == 1]
        # 遍历每个起飞点，提取路径
        for start_node in launch_nodes:
            temp = []  # 存储当前飞行路径
            current_node = start_node
            temp.append(current_node)  # 将起点添加到路径中
            # 提取从 start_node 到某个回收点的路径
            while True:
                next_node = None
                # 找到下一个节点
                for j in range(1, data.node_num - 1):
                    if j == current_node:
                        continue
                    if self.Y_drone[current_node][j] == 1:
                        next_node = j
                        break
                # 如果没有找到下一个节点，抛出异常
                if next_node is None:
                    raise ValueError(f"无法找到从节点 {current_node} 到回收点的路径")
                # 将下一个节点添加到路径中
                temp.append(next_node)
                current_node = next_node
                # 如果当前节点是回收点，结束当前飞行的路径提取
                if current_node in retrieve_nodes:
                    self.route_UAV.append(temp)
                    break
        print("\n\n 最优目标值：%.2f " % self.ObjVal)
        print("\n\n ------ 卡车的路径 ------- ")
        print('[', end='')
        for i in range(len(self.route_truck)):
            print(" %d " % self.route_truck[i], end=" ")
        print(']')
        print("\n\n ------ 无人机的路径 ------- ")
        for route in self.route_UAV:
            print('[', end='')
            for i in range(0, len(route)):
                print(" %d " % route[i], end=" ")
            print(']')
    def roll(self):
        """
               滚动优化：仅在服务失败时触发模型再求解，之前的前缀决策都冻结
               """
        idx = 0
        cout=0
        self.copy_from_solution()
        while idx < len(self.sequence):
            cid = self.sequence[idx]
            ok = self.simulate_availability(cid)
            self.simulated[cid] = ok
            self.frozen_customers.append(cid)
            if not ok:
                cout += 1
                self.failed_customers.append(cid)
                #  删除该客户的 Z[cid] == 1 约束
                self.model_build.remove_Z_constraint_for_customer(cid)
                #  打印约束
                for cid, constr in self.model_build.Z_constraints.items():
                    print(f"客户 {cid} 的 Z 约束名: {constr.ConstrName}, 表达式: {constr}")
                self.freeze_prefix()
                self.model.optimize()
                if self.model.status == GRB.OPTIMAL:
                    print("模型求解成功！")
                elif self.model.status == GRB.INFEASIBLE:
                    print("模型不可行，正在计算IIS...")
                    self.model.computeIIS()
                    self.model.write("infeasible_model.ilp")
                else:
                    print(f"模型状态: {self.model.status}")
                self.extract_solution(self.model, self.data)
                # 重新排序后的 sequence
                self.sequence = sorted(
                    range(1, self.customerNum + 1),
                    key=lambda cid: self.T_service[cid]
                )
            idx += 1
        print(f"\n---------------- 服务失败次数 cout = {cout} ----------------\n")
        print(self.failed_customers)

if __name__ == "__main__":
    # 1. 读取数据
    data = Data()
    path_customer = 'D:\\python\\pythonProject\\c102.csv'
    path_caf      = 'D:\\python\\pythonProject\\CAF.csv'
    customer_num = 16
    data.read_data(path_customer, path_caf, customer_num)
    data.print_data(customer_num)

    # 2. 建模并求解一次初始模型（获取初始解）
    model_handler = Model_builder()
    model_handler.build_model(data=data, solve_model=True)

    # 3. 获取初始解并可视化
    solution = Solution()
    solution.get_solution(data, model_handler.model)
    file_name = str(customer_num) + '_customer.pdf'
    solution.plot_solution(file_name=file_name, customer_num=customer_num)

    #  4. 嵌入滚动时域控制器
    print("\nRolling Controller 开始动态再优化...\n")
    controller = RollingController(data_obj=data,
                                    model_builder_obj=model_handler,
                                    solution_obj=solution, customerNum=customer_num)
    controller.roll()  # 启动滚动优化过程

    # 5. （可选）再次绘图或输出更新后的解
    updated_file = str(customer_num) + '_customer_updated.pdf'
    solution.plot_solution(file_name=updated_file, customer_num=customer_num)
