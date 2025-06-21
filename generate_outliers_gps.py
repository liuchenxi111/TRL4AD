import datetime
import math
import os
from datetime import timedelta

import numpy as np

from config import args


# Trajectory location offset
def perturb_point(point, level, offset=None):#point 是当前点的坐标和时间，level 是偏移级别，offset 是一个可选参数，用于指定偏移的方向。
    point, times = point[0], point[1]#从输入参数 point 中解包点的坐标和时间。point[0] 是点的坐标，point[1] 是时间。
    x, y = int(point // map_size[1]), int(point % map_size[1])#将点的坐标转换为二维坐标 (x, y)。这里使用了整除和取余操作符来分别获取 x 和 y 坐标。

    if offset is None:#如果没有提供 offset 参数，将使用一个预定义的偏移列表。然后随机选择一个偏移，并将其分配给 x_offset 和 y_offset。
        offset = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
        x_offset, y_offset = offset[np.random.randint(0, len(offset))]

    else:#如果提供了 offset 参数，直接使用它。
        x_offset, y_offset = offset

    if 0 <= x + x_offset * level < map_size[0] and 0 <= y + y_offset * level < map_size[1]:#检查新的 x 和 y 坐标是否在地图的边界内。如果是，应用偏移。
        x += x_offset * level
        y += y_offset * level

    return [int(x * map_size[1] + y), times]
# 📌 **空间扰动函数**（修改网格 & GPS 轨迹）
def perturb_point_gps(point_grid, point_gps, level, offset=None):
    """
    对网格轨迹和 GPS 轨迹同时进行扰动，确保两者偏移一致。

    参数：
    - point_grid: (grid_id, time)  网格轨迹点
    - point_gps: ((lat, lon), time)  GPS 轨迹点
    - level: 扰动级别
    - offset: (dx, dy) 偏移方向，例如 [0,1] 表示向右移动一个网格

    返回：
    - 新的 (grid_id, time)
    - 新的 ((lat, lon), time)
    """

    # **提取网格 & GPS 轨迹点信息**
    grid_id, time_grid = point_grid
    (lat, lon), time_gps = point_gps

    # **将网格索引转换为 x, y 坐标**
    x, y = int(grid_id // map_size[1]), int(grid_id % map_size[1])

    # **如果没有指定偏移，随机选择一个方向**
    if offset is None:
        offset_choices = [[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, -1], [-1, 1], [1, -1]]
        x_offset, y_offset = offset_choices[np.random.randint(0, len(offset_choices))]
    else:
        x_offset, y_offset = offset

    # **计算新的 x, y 坐标**
    new_x = x + x_offset * level
    new_y = y + y_offset * level

    # **确保新坐标仍然在地图范围内**
    if 0 <= new_x < map_size[0] and 0 <= new_y < map_size[1]:
        # **更新网格轨迹点**
        new_grid_id = int(new_x * map_size[1] + new_y)
        new_point_grid = [new_grid_id, time_grid]

        # **更新 GPS 轨迹点**
        new_lat = lat + x_offset * level * GRID_SIZE
        new_lon = lon + y_offset * level * GRID_SIZE
        new_point_gps = [(new_lat, new_lon), time_gps]

        return new_point_grid, new_point_gps

    # **如果超出边界，返回原始点**
    return point_grid, point_gps

def convert(point):#point 是点的线性坐标。
    x, y = int(point // map_size[1]), int(point % map_size[1])#将线性坐标转换为二维坐标 (x, y)。
    return [x, y]#


def distance(a, b):#a 和 b 是两个点的二维坐标。
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)#计算并返回两点之间的欧几里得距离。


def time_calcuate(vec, s):#vec 是一个包含日期和时间的向量，s 是要添加的秒数。
    a = datetime.datetime(vec[3], vec[4], vec[5], vec[0], vec[1], vec[2])#使用 vec 中的日期和时间创建一个 datetime 对象。
    t = a + timedelta(seconds=s)#将 timedelta 对象添加到 a 上，以计算新的时间。
    return [t.hour, t.minute, t.second, t.year, t.month, t.day]#返回新时间的小时、分钟、秒、年、月和日。


# Trajectory time offset
def perturb_time(traj, st_loc, end_loc, time_offset, interval):#traj 是轨迹数据，st_loc 和 end_loc 分别是时间偏移的起始和结束位置，time_offset 是时间偏移量，interval 是时间间隔。
    for i in range(st_loc, end_loc):
        traj[i][1] = time_calcuate(traj[i][1], int((i - st_loc + 1) * time_offset * interval))#遍历从 st_loc 到 end_loc 的轨迹点，对每个点的时间应用偏移。

    for i in range(end_loc, len(traj)):
        traj[i][1] = time_calcuate(traj[i][1], int((end_loc - st_loc) * time_offset * interval))#遍历从 end_loc 到轨迹末尾的点，对每个点的时间应用相同的偏移量。
    return traj

def perturb_batch(batch_x, level, prob, selected_idx):#batch_x 是一批轨迹数据，level 是偏移级别，prob 是异常点的概率，selected_idx 是选中的轨迹索引。
    noisy_batch_x = []#初始化一个空列表，用于存储带有噪声的轨迹数据。

    if args.dataset == 'porto':#根据数据集的类型设置时间间隔 interval。
        interval = 15
    else:
        interval = 10

    for idx, traj in enumerate(batch_x):#遍历每条轨迹。

        anomaly_len = int(len(traj) * prob)#计算异常长度，并随机选择异常的起始位置。
        anomaly_st_loc = np.random.randint(1, len(traj) - anomaly_len - 1)

        if idx in selected_idx:#如果当前轨迹被选中，计算异常的结束位置。
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            p_traj = traj[:anomaly_st_loc] + [perturb_point(p, level) for p in
                                              traj[anomaly_st_loc:anomaly_ed_loc]] + traj[anomaly_ed_loc:]#对选中的轨迹应用空间偏移。

            dis = max(distance(convert(traj[anomaly_st_loc][0]), convert(traj[anomaly_ed_loc][0])), 1)#计算异常段的起始点和结束点之间的距离，并根据距离计算时间偏移量。
            time_offset = (level * 2) / dis

            p_traj = perturb_time(p_traj, anomaly_st_loc, anomaly_ed_loc, time_offset, interval)#对轨迹应用时间偏移。

        else:
            p_traj = traj#如果轨迹没有被选中，保持原样。

        p_traj = p_traj[:int(len(p_traj) * args.obeserved_ratio)]#根据观察比例裁剪轨迹。
        noisy_batch_x.append(p_traj)#将处理后的轨迹添加到噪声轨迹列表中。

    return noisy_batch_x# return noisy_batch_x
# 📌 **批量扰动**
def perturb_batch_gps(batch_grid, batch_gps, level, prob, selected_idx):
    """
    对整个 batch 的网格轨迹 & GPS 轨迹进行扰动

    参数：
    - batch_grid: **网格轨迹**
    - batch_gps: **GPS 轨迹**
    - level: **扰动级别**
    - prob: **异常概率**
    - selected_idx: **选中进行扰动的轨迹索引**
    """
    noisy_batch_grid, noisy_batch_gps = [], []

    # 设置时间间隔
    interval = 15 if args.dataset == 'porto' else 10

    for idx, (traj_grid, traj_gps) in enumerate(zip(batch_grid, batch_gps)):
        anomaly_len = int(len(traj_grid) * prob)  # 计算扰动轨迹的长度
        anomaly_st_loc = np.random.randint(1, len(traj_grid) - anomaly_len - 1)

        if idx in selected_idx:
            anomaly_ed_loc = anomaly_st_loc + anomaly_len

            # **应用空间扰动**
            perturbed_grid, perturbed_gps = [], []
            for i, (p_grid, p_gps) in enumerate(zip(traj_grid, traj_gps)):
                if anomaly_st_loc <= i < anomaly_ed_loc:
                    new_grid, new_gps = perturb_point_gps(p_grid, p_gps, level)
                    perturbed_grid.append(new_grid)
                    perturbed_gps.append(new_gps)
                else:
                    perturbed_grid.append(p_grid)
                    perturbed_gps.append(p_gps)

            # **应用时间扰动**
            dis = max(math.sqrt((traj_gps[anomaly_st_loc][0][0] - traj_gps[anomaly_ed_loc][0][0]) ** 2 +
                                (traj_gps[anomaly_st_loc][0][1] - traj_gps[anomaly_ed_loc][0][1]) ** 2), 1)
            time_offset = (level * 2) / dis

            perturbed_grid = perturb_time(perturbed_grid, anomaly_st_loc, anomaly_ed_loc, time_offset, interval)
            perturbed_gps = perturb_time(perturbed_gps, anomaly_st_loc, anomaly_ed_loc, time_offset, interval)
        else:
            perturbed_grid = traj_grid
            perturbed_gps = traj_gps

        # noisy_batch_grid.append(perturbed_grid)
        # noisy_batch_gps.append(perturbed_gps)
        observed_len = int(len(perturbed_grid) * args.obeserved_ratio)
        noisy_batch_grid.append(perturbed_grid[:observed_len])
        noisy_batch_gps.append(perturbed_gps[:observed_len])

    return noisy_batch_grid, noisy_batch_gps

def generate_outliers(trajs, ratio=args.ratio, level=args.distance, point_prob=args.fraction):#接受一个参数 trajs 和三个可选参数，用于生成带有异常值的轨迹数据。
    traj_num = len(trajs)
    selected_idx = np.random.randint(0, traj_num, size=int(traj_num * ratio))#计算轨迹数量，并随机选择一些轨迹索引作为异常值。
    new_trajs = perturb_batch(trajs, level, point_prob, selected_idx)#对选中的轨迹应用空间和时间偏移。
    return new_trajs, selected_idx

def generate_outliers_gps(data_grid, data_gps, ratio=args.ratio, level=args.distance, point_prob=args.fraction):
    """
    生成带有异常值的轨迹数据（网格 & GPS）

    参数：
    - data_grid: **原始网格轨迹**
    - data_gps: **原始 GPS 轨迹**
    - ratio: **被扰动的轨迹比例**
    - level: **扰动级别**
    - point_prob: **轨迹点被扰动的概率**

    返回：
    - `outliers_trajs_grid`: 经过扰动的 **网格轨迹**
    - `outliers_trajs_gps`: 经过扰动的 **GPS 轨迹**
    - `outliers_idx`: 发生扰动的轨迹索引
    """
    traj_num = len(data_grid)
    selected_idx = np.random.randint(0, traj_num, size=int(traj_num * ratio))

    outliers_trajs_grid, outliers_trajs_gps = perturb_batch_gps(data_grid, data_gps, level, point_prob, selected_idx)

    return outliers_trajs_grid, outliers_trajs_gps, selected_idx

if __name__ == '__main__':
    np.random.seed(1234)#设置 NumPy 的随机数生成器的种子为 1234，以确保每次运行代码时生成的随机数是相同的。
    print("=========================")
    print("Dataset: " + args.dataset)
    print("d = {}".format(args.distance) + ", " + chr(945) + " = {}".format(args.fraction) + ", "
          + chr(961) + " = {}".format(args.obeserved_ratio))
    #打印分隔线和一些参数信息，包括数据集名称、距离参数 d、异常点概率 φ（phi）、观察比例 θ（theta）。
    if args.dataset == 'porto':
        map_size = (51, 119)#根据数据集的名称设置地图大小 map_size。
    elif args.dataset == 'cd':
        map_size = (167, 154)
    # 网格大小
    GRID_SIZE = 0.05  # 每个网格对应的实际 GPS 坐标偏移
    if args.dataset == 'porto':
        data_grid = np.load("/mnt/mydisk6/lcx666/mstoatd4/mstoatd-main/data/{}/test_trajs_init.npy".format(args.dataset), allow_pickle=True)#使用 np.load 函数加载指定数据集的初始测试数据。
        data_gps = np.load("/mnt/mydisk6/lcx666/mstoatd4/mstoatd-main/data/{}/test_data_gps_init_raw.npy".format(args.dataset), allow_pickle=True)
    else:
        data_grid = np.load("/mnt/mydisk6/lcx666/mstoatd4/mstoatd-main/data/{}/test_grid_init.npy".format(args.dataset),
                        allow_pickle=True)  # 使用 np.load 函数加载指定数据集的初始测试数据。
        data_gps = np.load(
        "/mnt/mydisk6/lcx666/mstoatd4/mstoatd-main/data/{}/test_gps_init.npy".format(args.dataset),
        allow_pickle=True)
    outliers_trajs_grid, outliers_trajs_gps, outliers_idx = generate_outliers_gps(data_grid,data_gps)#将带有异常值的轨迹数据和索引转换为 NumPy 数组。
    outliers_trajs_grid = np.array(outliers_trajs_grid, dtype=object)
    outliers_trajs_gps = np.array(outliers_trajs_gps, dtype=object)
    outliers_idx = np.array(outliers_idx)
    #使用 np.save 函数将带有异常值的轨迹数据和索引保存到文件中。
    np.save("/mnt/mydisk6/lcx666/mstoatd4/mstoatd-main/data/{}/outliers_data_init_grid_{}_{}_{}.npy".format(args.dataset, args.distance, args.fraction,
                                                               args.obeserved_ratio), outliers_trajs_grid)
    np.save("/mnt/mydisk6/lcx666/mstoatd4/mstoatd-main/data/{}/outliers_data_init_gps_{}_{}_{}.npy".format(args.dataset,args.distance,args.fraction,
                                                               args.obeserved_ratio),outliers_trajs_gps)
    np.save("/mnt/mydisk6/lcx666/mstoatd4/mstoatd-main/data/{}/outliers_idx_init_gps_{}_{}_{}.npy".format(args.dataset, args.distance, args.fraction,
                                                              args.obeserved_ratio), outliers_idx)
    


    ########删掉了porto排序
    ########删掉了cd排序