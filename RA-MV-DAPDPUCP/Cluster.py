import numpy as np
import copy
from tqdm import tqdm

class ClusterInfo:
    def __init__(self, cluster_id, indices, center, total_demand):
        self.cluster_id = cluster_id  # 聚类编号
        self.indices = indices  # 该聚类中的所有数据点的索引
        self.center = center
        self.total_demand = total_demand  # 该聚类的总需求

    def __repr__(self):
        return f"ClusterInfo(cluster_id={self.cluster_id}, total_demand={self.total_demand}, indices={self.indices})"

class FCM:
    def __init__(self, data, clust_num, iter_num, vehicle_capacity, customers, drone_weight, remain_demand):
        # 客户信息  仅包含 [customer.cust_no, customer.xcoord, customer.ycoord, customer.demand, customer.start_time, customer.end_time,customer.drone_eligible
        self.data = data
        self.cnum = clust_num                           # 聚类数目
        self.sample_num=data.shape[0]                   # 样本数量
        self.dim = 2                                    # 数据维度   仅取位置坐标信息x与y
        self.vehicle_capacity = vehicle_capacity
        self.customers = customers                      #客户全部信息 存储客户实例的列表
        self.drone_weight = drone_weight
        self.remain_demand=remain_demand
        self.m = 2                                      # 模糊系数
        self.dict_cluster = []                          # 分类后的客户点
        self.best_J=10000000
        self.best_U=None
        self.best_C=None
        self.clusters = []                              # 存储聚类信息的列表
        self.iter_num_COUT = 0
        Jlist=[]                                        # 存储目标函数值的列表
        U = self.Initial_U(self.sample_num, self.cnum)  # 初始化隶属度矩阵 U
        C = self.Cen_Iter(self.data, U, self.cnum)      # 计算类中心
        self.label = np.argmax(U, axis=0)               # 所有样本的分类标签
        self.Initial_Clusters(C)                        # 初始化聚类信息
        for i in range(iter_num):                       # 迭代次数
            C = self.Cen_Iter(self.data, U, self.cnum)  # 计算类中心
            U = self.U_Iter(U, C)                       # 更新隶属度矩阵 U
            self.label = np.argmax(U, axis=0)           # 所有样本的分类标签
            self.update_clusters(C)                     # 更新聚类
            if self.check_clusters_demand()==0:
                U,C = self.find_best_transfer(U, C)     #返回值
            print("第%d次迭代" %(i+1) ,end="")
            print("聚类中心",C)
            J = self.J_calcu(self.data, U, C)           # 计算目标函数值
            Jlist = np.append(Jlist, J)                 # 将目标函数值添加到列表中
            if J<self.best_J:
                self.best_J=J
                self.best_U = copy.deepcopy(U)
                self.best_C = copy.deepcopy(C)
                self.iter_num_COUT=i
        print("最佳目标值:",self.best_J )
        self.label = np.argmax(self.best_U, axis=0)     # 所有样本的分类标签
        self.Clast = self.best_C                        # 最终的类中心矩阵
        self.update_clusters(self.best_C)
        for i in range(len(self.label)):
            self.customers[i].cluster = self.label[i]
        self.Jlist = Jlist  # 存储目标函数值的列表

    # 初始化隶属度矩阵U
    def Initial_U(self, sample_num, cluster_n):
        U = np.random.rand(sample_num, cluster_n)       # 随机初始化隶属度矩阵U
        row_sum = np.sum(U, axis=1)                     # 按行求和
        row_sum = 1 / row_sum                           # 每行元素和取倒数
        U = np.multiply(U.T, row_sum)             # 确保每列和为1
        return U                                        # 返回初始化的隶属度矩阵

    # 计算类中心
    def Cen_Iter(self, data, U, cluster_n):
        c_new = np.empty(shape=[0, self.dim])                   # 初始化新的类中心矩阵
        for i in range(0, cluster_n):                           # 对每一个类别
            u_ij_m = U[i, :] ** self.m                          # 计算隶属度的m次方
            sum_u = np.sum(u_ij_m)                              # 求和
            ux = np.dot(u_ij_m, data[:,1:3])                    # 计算加权平均值
            ux = np.reshape(ux, (1, self.dim))            # 调整形状
            c_new = np.append(c_new, ux /sum_u, axis=0)         # 添加新的类中心到类中心矩阵
        return c_new  # 返回更新后的类中心矩阵

    # 更新隶属度矩阵U
    def U_Iter(self, U, c):
        for i in range(0, self.cnum):  # 对每一个类别
            for j in range(0, self.sample_num):  # 对每一个样本
                Asum = 0
                for k in range(0, self.cnum):  # 对每一个类别
                    temp = (np.linalg.norm(self.data[j, 1:3] - c[i, :]) /
                            np.linalg.norm(self.data[j, 1:3] - c[k, :])) ** (2 / (self.m - 1))  # 计算更新公式
                    Asum = temp + Asum  # 求和
                U[i, j] = 1 / Asum  # 更新隶属度矩阵U
        return U  # 返回更新后的隶属度矩阵 U

    # 初始化聚类信息
    def Initial_Clusters(self, C):
        for type in range(self.cnum):  # 对每一个类型
            index = [x for x in range(len(self.label)) if self.label[x] == type]  # 获取索引
            total_demand = sum(self.data[idx][3] for idx in index if self.data[idx][3]>0)
            type_center=C[type]
            cluster_info = ClusterInfo(type, index, type_center, total_demand)
            self.clusters.append(cluster_info)
        return

    # 更新聚类信息
    def update_clusters(self,C):
        """每次迭代后更新已有的聚类信息"""
        for type in range(self.cnum):  # 对每一个类型
            index = [x for x in range(len(self.label)) if self.label[x] == type]  # 获取索引
            total_demand = sum(self.data[idx][3] for idx in index if self.data[idx][3]>0)
            type_center=C[type]
            self.clusters[type].indices=index
            self.clusters[type].total_demand = total_demand
            self.clusters[type].center = type_center
        return

    def check_clusters_demand(self):
        for cluster in self.clusters:
            if cluster.total_demand > self.vehicle_capacity-self.drone_weight:
                return 0
        return 1

    # 找到该聚类中的候选客户进行转移
    def find_best_transfer(self, U, C):
        cout=0
        # 假设每次更新时，首先我们检查哪些聚类中的客户需要被转移
        candidates_to_transfer = []         # 存储需要转移客户的聚类
        best_transfer = ()                  # 初始化最优转移方案
        # 1. 更新需要转移的客户的聚类            ( 根据某个条件，比如需求量不平衡等 )
        for cluster in self.clusters:
            if cluster.total_demand > self.vehicle_capacity-self.drone_weight-self.remain_demand:
                candidates_to_transfer.append(cluster)
        # 2. 对于每个需要转移的聚类，选择最优的客户进行转移
        for from_cluster in candidates_to_transfer:
            while from_cluster.total_demand>self.vehicle_capacity-self.drone_weight-self.remain_demand:
                best_J = 100000.0
                for to_cluster in self.clusters:
                    if from_cluster.cluster_id == to_cluster.cluster_id:    # 避免选择 相同的聚类
                        continue
                    if to_cluster.total_demand>self.vehicle_capacity-self.drone_weight-self.remain_demand:      # 避免选择容量超出的聚类
                        continue
                    for customer_idx in tqdm(from_cluster.indices, desc="检查每个客户", ncols=100, leave=False):
                        if self.customers[customer_idx].demand<0:           # 避免选择需求为负的客户
                            continue
                        if to_cluster.total_demand+self.customers[customer_idx].demand>self.vehicle_capacity-self.drone_weight-self.remain_demand:
                            continue
                        prev_U = U.copy()  # 备份U
                        tem = U[from_cluster.cluster_id, customer_idx]
                        U[from_cluster.cluster_id, customer_idx] = U[to_cluster.cluster_id, customer_idx]    # 分配到最近的聚类
                        U[to_cluster.cluster_id, customer_idx] = tem
                        C = self.Cen_Iter(self.data, U, self.cnum)                                  # 计算类中心
                        J = self.J_calcu(self.data, U, C)                                           # 计算目标函数值
                        if J < best_J:
                            best_J = J
                            best_transfer = (from_cluster.cluster_id, to_cluster.cluster_id, customer_idx)
                        U = prev_U
                if best_transfer:
                    tem = U[best_transfer[0], best_transfer[2]]
                    U[best_transfer[0], best_transfer[2]] = U[best_transfer[1], best_transfer[2]]  # 分配到最近的聚类
                    U[best_transfer[1], best_transfer[2]] = tem
                    self.label = np.argmax(U, axis=0)  # 所有样本的分类标签
                    C = self.Cen_Iter(self.data, U, self.cnum)  # 计算类中心
                    self.update_clusters(C)
                    best_transfer = ()
                else:
                    U = self.Initial_U(self.sample_num, self.cnum)  # 初始化隶属度矩阵 U
                    C = self.Cen_Iter(self.data, U, self.cnum)  # 计算类中心
                    self.label = np.argmax(U, axis=0)  # 所有样本的分类标签
                    self.update_clusters(C)
                    for cluster in self.clusters:
                        if cluster.total_demand > self.vehicle_capacity - self.drone_weight:
                            candidates_to_transfer.append(cluster)
                    print(f"未找到最优转移，跳过更新")
        return U, C

    # 计算目标函数值
    def J_calcu(self, data, U, c):
        temp1 = np.zeros(U.shape)               # 初始化临时矩阵
        for i in range(0, U.shape[0]):          # 对每一个类别
            for j in range(0, U.shape[1]):      # 对每一个样本
                temp1[i, j] = (np.linalg.norm(data[j, 1:3] - c[i, :])) ** 2 * U[i, j] ** self.m  # 计算目标函数值
        J = np.sum(np.sum(temp1))               # 求和
        return J                                # 返回目标函数值

    def print_clusters(self):
        """
        打印所有聚类信息
        """
        for cluster in self.clusters:
            print(cluster)