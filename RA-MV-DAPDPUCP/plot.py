import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def ensure_all_points_connected(points, edges, cluster_center):
    """
    确保所有点都被正确连接到边界
    :param points: 聚类的所有点 (N, 2)
    :param edges: 当前 Alpha Shape 的边界线段
    :param cluster_center: 当前聚类的中心点 (x, y)
    :return: 更新后的边界线段
    """
    if points is None or points.size == 0 or edges is None:
        print("Warning: Points or edges are empty in ensure_all_points_connected.")
        return edges  # 如果边为空，直接返回空集合
    # 提取已连接的点
    connected_points = set([tuple(edge[0]) for edge in edges] + [tuple(edge[1]) for edge in edges])
    # 寻找未连接的点
    unconnected_points = [point for point in points if tuple(point) not in connected_points]
    updated_edges = list(edges)
    for point in unconnected_points:
        # 找到距离最近的边界点
        nearest_point = None
        min_distance = float('inf')
        for edge in edges:
            for connected_point in edge:  # 仅在边界线上找最近点
                dist = np.linalg.norm(point - np.array(connected_point))
                if dist < min_distance:
                    nearest_point = connected_point
                    min_distance = dist
            # 添加新的连接
        if nearest_point is not None:
           updated_edges.append((point, np.array(nearest_point)))

        return updated_edges

def alpha_shape_with_corrections(points, alpha, cluster_center):
    """
    计算点集的 Alpha Shape (凹形边界)
    :param points: 点集 (N, 2)
    :param alpha: 控制形状的参数，越小越贴近点的分布
    :return: Alpha Shape 的边界线段
    """
    if len(points) < 4:  # 点数少于4时，直接返回凸包
        hull = ConvexHull(points)
        return [(points[simplex[0]], points[simplex[1]]) for simplex in hull.simplices]

    tri = Delaunay(points)  # 构建 Delaunay 三角剖分
    edges = set()
    edge_count = {}

    # 遍历所有三角形
    for ia, ib, ic in tri.simplices:
        # 获取三角形的顶点
        pa, pb, pc = points[ia], points[ib], points[ic]
        # 计算三角形的外接圆半径
        a, b, c = np.linalg.norm(pa - pb), np.linalg.norm(pb - pc), np.linalg.norm(pc - pa)
        s = (a + b + c) / 2.0  # 半周长
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))  # 三角形面积
        circ_radius = a * b * c / (4.0 * area)  # 外接圆半径
        if circ_radius < 1.0 / alpha:  # 如果外接圆的半径小于阈值
            # 添加边到集合，并统计边的出现次数
            for edge in [(ia, ib), (ib, ic), (ic, ia)]:
                edge = tuple(sorted(edge))  # 确保边的顺序一致
                edge_count[edge] = edge_count.get(edge, 0) + 1

    # 仅保留出现一次的边（即外部边界的边）
    for edge, count in edge_count.items():
        if count == 1:
            edges.add(edge)

    # 转换为实际坐标
    boundary_edges = [(points[edge[0]], points[edge[1]]) for edge in edges]
    # 确保所有点都被包含在边界中
    boundary_edges = ensure_all_points_connected(points, boundary_edges, cluster_center)
    return boundary_edges

def plot_clus(FCMRes):

    mark = ['or', 'ob', 'og', 'om', 'oy', 'oc', 'sr', 'sb', 'sg', 'sm', 'sy', 'sc']  # 定义不同标记的颜色

    # 第1张图（聚类后结果图）
    plt.subplot(121)  # 创建第1个子图
    j = 0
    for type in FCMRes.label:
        plt.plot(FCMRes.data[j:j + 1, 1], FCMRes.data[j:j + 1, 2], mark[type % 12], markersize=2)  # 根据类别绘制数据点
        j += 1
    plt.plot(FCMRes.Clast[:, 0], FCMRes.Clast[:, 1], 'k*', markersize=4)  # 绘制聚类中心点
    for cluster in FCMRes.clusters:
        cluster_data = FCMRes.data[cluster.indices]
        cluster_center = FCMRes.Clast[cluster.cluster_id]
        if len(cluster_data) > 3:  # 至少需要4个点计算 α形状
            alpha = 0.1  # 控制边界贴合度的参数，可以根据需要调整
            edges = alpha_shape_with_corrections(cluster_data[:, 1:3], alpha, cluster_center)
            if edges is None or len(edges) == 0:  # 如果边界为空
                print(f"Warning: No edges for cluster {cluster.cluster_id}")
                continue

            for edge in edges:
                plt.plot(
                    [edge[0][0], edge[1][0]],  # x 坐标
                    [edge[0][1], edge[1][1]],  # y 坐标
                    'k-', lw=1
                )
        else:
            # 点数少时直接用凸包
            hull = ConvexHull(cluster_data[:, 1:3])
            plt.plot(
                np.append(cluster_data[hull.vertices, 1], cluster_data[hull.vertices[0], 1]),
                np.append(cluster_data[hull.vertices, 2], cluster_data[hull.vertices[0], 2]),
                'k-', lw=1
            )
    plt.xlabel('x')  # 设置x轴标签
    plt.ylabel('y')  # 设置y轴标签
    plt.title("聚类后结果")  # 设置图表标题


    plt.subplot(122)  # 创建第2个子图
    xxx = np.arange(0, 30, 1)  # 设置x轴数据
    plt.plot(xxx, FCMRes.Jlist, 'g-')  # 绘制聚类目标函数值随迭代次数变化曲线
    # 设置 x 轴显示的范围和刻度
    plt.xlim(0, 30)  # 设置 x 轴显示的范围
    plt.xticks(np.arange(0, 31, 1))  # 设置 x 轴的刻度
    plt.xlabel('迭代次数', fontproperties="SimHei")  # 设置x轴标签
    plt.ylabel('聚类目标函数值', fontproperties="SimHei")  # 设置y轴标签
    plt.title("聚类方案调整图")  # 设置图表标题
    plt.subplots_adjust(wspace=0.5, hspace=0.5)  # 增大水平和垂直间距

import matplotlib.pyplot as plt


def plot_truck(TRUCK_Routes, customers):
    # 设置图像大小
    plt.figure(figsize=(10, 8))  # 设置宽度为10英寸，高度为8英寸
    # 颜色列表
    colors = ['r', 'b', 'g', 'purple' , 'y', 'c', 'orange', 'm', 'brown', 'pink', 'lime', 'teal']
    # 起点 depot 坐标
    depot = [0, 0, 0, 0, 0, 2000, -1]  # 起点坐标
    depot_x, depot_y = depot[1], depot[2]  # 0, 0 为起点坐标

    # 生成卡车路径
    for idx, truck in enumerate(TRUCK_Routes):
            path_id=truck.vehicle_id                # 获取卡车ID
            route = truck.Troute                    # 获取该卡车的路径
            path_x, path_y = [depot_x], [depot_y]   # 将起点加入路径
            for cust_no in route:
                # 查找每个客户对象
                customer = next((cust for cust in customers if cust.cust_no == cust_no), None)
                if customer:  # 如果找到对应客户
                    path_x.append(customer.xcoord)  # 添加客户的x坐标
                    path_y.append(customer.ycoord)  # 添加客户的y坐标
                    # 在每个客户节点的坐标上方显示客户的序号
                    plt.text(customer.xcoord, customer.ycoord + 2, str(customer.cust_no),
                             fontsize=10, ha='center', color='black')  # 序号显示在节点上方
            path_x.append(depot_x)  # 回到起点
            path_y.append(depot_y)  # 回到起点
            # 获取当前路径的颜色，从mark列表中获取
            path_color = colors[idx % len(colors)]  # 如果路径数量超过mark列表长度，循环使用颜色
            plt.plot(path_x, path_y, label=f'路径 {path_id}', linewidth=2, marker='o', markersize=4, color=path_color)  # 绘制路径
            plt.plot(path_x[1], path_y[1], label=f'路径 {path_id}', linewidth=2, marker='^', markersize=8, color=path_color)          # 重新绘制第一个客户节点


    plt.xlabel('x')  # 设置x轴标签
    plt.ylabel('y')  # 设置y轴标签
    plt.title('卡车路径规划图')  # 设置图表标题
    plt.show()


def plot_drone_routes(TRUCK_Routes, DRONE_Routes, customers):
    # 设置图像大小
    plt.figure(figsize=(10, 8))  # 设置宽度为10英寸，高度为8英寸
    # 颜色列表
    colors = ['r', 'b', 'g', 'purple', 'y', 'c', 'orange', 'm', 'brown', 'pink', 'lime', 'teal']
    # 起点 depot 坐标
    depot = [0, 0, 0, 0, 0, 2000, -1]  # 起点坐标
    depot_x, depot_y = depot[1], depot[2]  # 0, 0 为起点坐标

    # 生成路径
    for idx, truck in enumerate(TRUCK_Routes):
        path_id = truck.vehicle_id  # 获取卡车ID
        route = truck.Troute  # 获取该卡车的路径
        path_x, path_y = [depot_x], [depot_y]  # 将起点加入路径
        for cust_no in route:
            # 查找每个客户对象
            customer = next((cust for cust in customers if cust.cust_no == cust_no), None)
            if customer:  # 如果找到对应客户
                path_x.append(customer.xcoord)  # 添加客户的x坐标
                path_y.append(customer.ycoord)  # 添加客户的y坐标
                plt.text(customer.xcoord, customer.ycoord + 2, str(customer.cust_no),
                         fontsize=10, ha='center', color='black')  # 序号显示在节点上方
        path_x.append(depot_x)  # 回到起点
        path_y.append(depot_y)  # 回到起点
        # 获取当前路径的颜色，从mark列表中获取
        path_color = colors[idx % len(colors)]  # 如果路径数量超过mark列表长度，循环使用颜色
        plt.plot(path_x, path_y, label=f'路径 {path_id}', linewidth=2, marker='o', markersize=4,
                 color=path_color)  # 绘制路径
        plt.plot(path_x[1], path_y[1], label=f'路径 {path_id}', linewidth=2, marker='^', markersize=8,
                 color=path_color)  # 重新绘制第一个客户节点

    for drone_idx, drone in enumerate(DRONE_Routes):
        # 遍历每个无人机的路径
        for trip_idx, trip in enumerate(drone.route):
            path = trip['path']  # 获取该行程的路径
            launch_node = trip['launch_node']  # 起飞节点
            retrieval_node = trip['retrieval_node']  # 回收节点

            # 构建路径坐标
            path_x = []
            path_y = []

            # 添加起飞节点的坐标
            launch_customer = next((cust for cust in customers if cust.cust_no == launch_node), None)
            if launch_customer:
                path_x.append(launch_customer.xcoord)
                path_y.append(launch_customer.ycoord)

            # 遍历路径中的每个客户节点
            for cust_no in path:
                customer = next((cust for cust in customers if cust.cust_no == cust_no), None)
                if customer:
                    path_x.append(customer.xcoord)
                    path_y.append(customer.ycoord)
                    # 在每个客户节点的坐标上方显示客户的序号
                    plt.text(customer.xcoord, customer.ycoord + 2, str(customer.cust_no),
                             fontsize=10, ha='center', color='black')  # 序号显示在节点上方

            # 添加回收节点的坐标
            retrieval_customer = next((cust for cust in customers if cust.cust_no == retrieval_node), None)
            if retrieval_customer:
                path_x.append(retrieval_customer.xcoord)
                path_y.append(retrieval_customer.ycoord)

            # # 获取当前路径的颜色，从colors列表中获取
            # path_color = colors[(drone_idx * len(drone.route) + trip_idx) % len(colors)]  # 结合无人机和路径的索引获取颜色

            # 绘制路径（所有路径都用黑色虚线表示）
            plt.plot(path_x, path_y, label=f'无人机 {drone.vehicle_id} 路径 {trip_idx + 1}',
                     linewidth=2, linestyle='--', color='k')           # 设置黑色虚线

            # 绘制起飞节点
            plt.plot(path_x[0], path_y[0], 'go', markersize=8)   # 起飞点标记为绿色圆圈
            # 绘制回收节点
            plt.plot(path_x[-1], path_y[-1], 'ro', markersize=8)  # 回收点标记为红色圆圈

    # 设置图表标题和标签
    plt.xlabel('x')  # 设置x轴标签
    plt.ylabel('y')  # 设置y轴标签
    plt.title('卡车-无人机路径规划图')  # 设置图表标题
    plt.legend()  # 显示图例

    # 显示图形
    plt.show()

def plot_single_drone_routes(TRUCK_Routes, drone, customers):
    # 设置图像大小
    plt.figure(figsize=(10, 8))  # 设置宽度为10英寸，高度为8英寸
    # 颜色列表
    colors = ['r', 'b', 'g', 'purple', 'y', 'c', 'orange', 'm', 'brown', 'pink', 'lime', 'teal']
    # 起点 depot 坐标
    depot = [0, 0, 0, 0, 0, 2000, -1]  # 起点坐标
    depot_x, depot_y = depot[1], depot[2]  # 0, 0 为起点坐标

    # 生成路径
    for idx, truck in enumerate(TRUCK_Routes):
        path_id = truck.vehicle_id  # 获取卡车ID
        route = truck.Troute  # 获取该卡车的路径
        path_x, path_y = [depot_x], [depot_y]  # 将起点加入路径
        for cust_no in route:
            # 查找每个客户对象
            customer = next((cust for cust in customers if cust.cust_no == cust_no), None)
            if customer:  # 如果找到对应客户
                path_x.append(customer.xcoord)  # 添加客户的x坐标
                path_y.append(customer.ycoord)  # 添加客户的y坐标
                plt.text(customer.xcoord, customer.ycoord + 2, str(customer.cust_no),
                         fontsize=10, ha='center', color='black')  # 序号显示在节点上方
        path_x.append(depot_x)  # 回到起点
        path_y.append(depot_y)  # 回到起点
        # 获取当前路径的颜色，从mark列表中获取
        path_color = colors[idx % len(colors)]  # 如果路径数量超过mark列表长度，循环使用颜色
        plt.plot(path_x, path_y, label=f'路径 {path_id}', linewidth=2, marker='o', markersize=4,
                 color=path_color)  # 绘制路径
        plt.plot(path_x[1], path_y[1], label=f'路径 {path_id}', linewidth=2, marker='^', markersize=8,
                 color=path_color)  # 重新绘制第一个客户节点

    for trip_idx, trip in enumerate(drone.route):
        path = trip['path']  # 获取该行程的路径
        launch_node = trip['launch_node']  # 起飞节点
        retrieval_node = trip['retrieval_node']  # 回收节点

        # 构建路径坐标
        path_x = []
        path_y = []

        # 添加起飞节点的坐标
        launch_customer = next((cust for cust in customers if cust.cust_no == launch_node), None)
        if launch_customer:
            path_x.append(launch_customer.xcoord)
            path_y.append(launch_customer.ycoord)

        # 遍历路径中的每个客户节点
        for cust_no in path:
            customer = next((cust for cust in customers if cust.cust_no == cust_no), None)
            if customer:
                path_x.append(customer.xcoord)
                path_y.append(customer.ycoord)
                # 在每个客户节点的坐标上方显示客户的序号
                plt.text(customer.xcoord, customer.ycoord + 2, str(customer.cust_no),
                         fontsize=10, ha='center', color='black')  # 序号显示在节点上方

        # 添加回收节点的坐标
        retrieval_customer = next((cust for cust in customers if cust.cust_no == retrieval_node), None)
        if retrieval_customer:
            path_x.append(retrieval_customer.xcoord)
            path_y.append(retrieval_customer.ycoord)

        # # 获取当前路径的颜色，从colors列表中获取
        # path_color = colors[(drone_idx * len(drone.route) + trip_idx) % len(colors)]  # 结合无人机和路径的索引获取颜色

        # 绘制路径（所有路径都用黑色虚线表示）
        plt.plot(path_x, path_y, label=f'无人机 {drone.vehicle_id+1} 路径 {trip_idx + 1}',
                 linewidth=2, linestyle='--', color='k')           # 设置黑色虚线

        # 绘制起飞节点
        plt.plot(path_x[0], path_y[0], 'go', markersize=8)   # 起飞点标记为绿色圆圈
        # 绘制回收节点
        plt.plot(path_x[-1], path_y[-1], 'ro', markersize=8)  # 回收点标记为红色圆圈

    # 设置图表标题和标签
    plt.xlabel('x')  # 设置x轴标签
    plt.ylabel('y')  # 设置y轴标签
    plt.title('卡车-无人机路径规划图')  # 设置图表标题
    # 只显示无人机相关的图例
    drone_labels = [f'无人机 {drone.vehicle_id+1} 路径 {trip_idx + 1}'
                     for trip_idx, trip in enumerate(drone.route)]
    plt.legend(drone_labels, loc='upper left')  # 显示无人机图例

    # 显示图形
    plt.show()

