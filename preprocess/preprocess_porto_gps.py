import datetime
import json
import random

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split

from preprocess_utils import in_boundary,  grid_mapping, generate_matrix#cutting_trajs,

def cutting_trajs(traj, gps_traj, longest, shortest):
    cutted_trajs = []
    cutted_gps_trajs = []
    while len(traj) > longest:
        random_length = np.random.randint(shortest, longest)
        cutted_traj = traj[:random_length]
        cutted_gps_traj = gps_traj[:random_length]  # Ensure same slicing
        cutted_trajs.append(cutted_traj)
        cutted_gps_trajs.append(cutted_gps_traj)
        traj = traj[random_length:]
        gps_traj = gps_traj[random_length:]
    return cutted_trajs, cutted_gps_trajs
def time_convert(timestamp):
    return datetime.datetime.fromtimestamp(timestamp)


def preprocess(trajectories, traj_num, point_num):
    trajs = []  # Preprocessed trajectory sequences
    trajs_gps = []  # Preprocessed GPS trajectory sequences

    for traj in trajectories.itertuples():  # Iterate over the DataFrame rows
        traj_seq = []
        gps_seq = []
        valid = True  # Flag to determine whether a trajectory is within boundary

        polyline = json.loads(traj.POLYLINE)  # Convert JSON string to Python list
        timestamp = traj.TIMESTAMP  # Extract timestamp

        if len(polyline) >= shortest:  # Only process trajectories of sufficient length
            for lng, lat in polyline:
                if in_boundary(lat, lng, boundary):  # Check if point is within boundary
                    gps_i = lat
                    gps_j = lng

                    #gps_i = int((lat - boundary['min_lat'])*1000000)
                    #gps_j = int((lng - boundary['min_lng'])*1000000)
                    #print((gps_i, gps_j))
                    grid_i = int((lat - boundary['min_lat']) / lat_size)
                    grid_j = int((lng - boundary['min_lng']) / lng_size)
                    t = datetime.datetime.fromtimestamp(timestamp)
                    t = [t.hour, t.minute, t.second, t.year, t.month, t.day]  # Time vector

                    traj_seq.append([int(grid_i * lng_grid_num + grid_j), t])  # Append trajectory sequence
                    gps_seq.append([(gps_i, gps_j), t])  # Append corresponding GPS sequence
                    timestamp += 15  # Increment timestamp by 15 seconds
                else:
                    valid = False  # If a point is out of boundary, discard trajectory
                    break

            # Randomly delete 30% trajectory points to make the sampling rate not fixed
            to_delete = set(random.sample(range(len(traj_seq)), int(len(traj_seq) * 0.3)))
            traj_seq = [item for index, item in enumerate(traj_seq) if index not in to_delete]
            gps_seq = [item for index, item in enumerate(gps_seq) if index not in to_delete]

            # Lengths are limited from shortest to longest
            if valid:
                if len(traj_seq) <= longest and len(gps_seq) <= longest:
                    trajs.append(traj_seq)
                    trajs_gps.append(gps_seq)  # Ensure both are appended
                else:
                    cutted_trajs, cutted_gps_trajs = cutting_trajs(traj_seq, gps_seq, longest, shortest)
                    trajs += cutted_trajs
                    trajs_gps += cutted_gps_trajs

    traj_num += len(trajs)  # Update trajectory count
    for traj in trajs:
        point_num += len(traj)  # Update total point count

    return trajs, trajs_gps, traj_num, point_num  # Return processed trajectories



def main():
    # Read csv file
    trajectories = pd.read_csv("{}/{}.csv".format(data_dir, data_name), header=0, usecols=['POLYLINE', 'TIMESTAMP'])
    #使用 pandas 的 read_csv 函数从CSV文件中读取数据，文件路径由 data_dir 和 data_name 拼接生成。
    #usecols 参数指定只读取 POLYLINE（轨迹）和 TIMESTAMP（时间戳）两列数据。
    trajectories['datetime'] = trajectories['TIMESTAMP'].apply(time_convert)
    #将 TIMESTAMP 列中的时间戳转为日期时间格式，并存储在 datetime 列，time_convert 是一个将时间戳转换为日期时间格式的函数。

    # Inititial dataset  初始化开始时间为 2013年7月1日，结束时间为 2013年9月1日。
    start_time = datetime.datetime(2013, 7, 1, 0, 0, 0)
    end_time = datetime.datetime(2013, 9, 1, 0, 0, 0)

    traj_num, point_num = 0, 0#初始化两个计数器 traj_num 和 point_num，用于统计处理的轨迹和数据点的数量。

    # Select trajectories from start time to end time
    trajs = trajectories[(trajectories['datetime'] >= start_time) & (trajectories['datetime'] < end_time)]#选取在 start_time 和 end_time 之间的轨迹数据。
    # trajs = trajectories
    print("原始数据集数量：",len(trajs))
    preprocessed_trajs, preprocessed_trajs_gps,traj_num, point_num = preprocess(trajs, traj_num, point_num)#调用 preprocess 函数对轨迹数据进行预处理，并返回处理后的轨迹列表、更新的轨迹数量和数据点数量。
    #train_data, test_data = train_test_split(preprocessed_trajs, test_size=0.2, random_state=42)
    # 打印前五组轨迹数据
    print("First 5 preprocessed_trajs:")
    for i, traj in enumerate(preprocessed_trajs[:5]):
        print(f"Trajectory {i + 1}:", traj)

    print("\nFirst 5 preprocessed_trajs_gps:")
    for i, traj_gps in enumerate(preprocessed_trajs_gps[:5]):
        print(f"GPS Trajectory {i + 1}:", traj_gps)

    # 打印总的轨迹数和点数
    print("\nTotal number of trajectories:", traj_num)
    print("Total number of trajectory points:", point_num)
    train_trajs, test_trajs, train_trajs_gps, test_trajs_gps = train_test_split(
        preprocessed_trajs, preprocessed_trajs_gps, test_size=0.2, random_state=42)
    #使用 train_test_split 函数将预处理后的轨迹数据划分为 80% 的训练集和 20% 的测试集，random_state=42 保证划分的随机性一致。
    #np.save("../data/porto/train_data_init.npy", np.array(train_data, dtype=object))#将训练集和测试集数据保存为 .npy 格式，方便后续的加载和使用。
    #np.save("../data/porto/test_data_init.npy", np.array(test_data, dtype=object))
    # Save processed data
    np.save("../data/porto/train_trajs_init.npy", np.array(train_trajs, dtype=object))
    np.save("../data/porto/test_trajs_init.npy", np.array(test_trajs, dtype=object))
    np.save("../data/porto/train_data_gps_init_raw.npy", np.array(train_trajs_gps, dtype=object))
    np.save("../data/porto/test_data_gps_init_raw.npy", np.array(test_trajs_gps, dtype=object))

    start_time = datetime.datetime(2013, 9, 1, 0, 0, 0)




if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)
    #设定随机数生成器的种子，确保结果可复现。无论是 random 还是 numpy 生成的随机数，在每次运行时都会生成相同的结果。
    data_dir = '/mnt/mydisk6/lcx666/mstoatd4/mstoatd-main/data/porto'
    data_name = "porto"

    boundary = {'min_lat': 41.140092, 'max_lat': 41.185969, 'min_lng': -8.690261, 'max_lng': -8.549155}
    #boundary 定义了经纬度的边界（最小/最大纬度和经度），这些边界将用于确定轨迹点是否在定义的区域内。
    grid_size = 0.1#grid_size 定义了每个网格单元的大小，单位是经纬度差值。
    shortest, longest = 20, 50#shortest 和 longest 分别定义了轨迹的最短和最长长度。

    lat_size, lng_size, lat_grid_num, lng_grid_num = grid_mapping(boundary, grid_size)
    #grid_mapping 是一个函数，用于将经纬度边界转换为网格。该函数返回每个网格的大小以及在纬度和经度方向上分别划分的网格数量。
    A, D = generate_matrix(lat_grid_num, lng_grid_num)
    #generate_matrix 是另一个函数，负责生成两个矩阵 A 和 D：A 通常是邻接矩阵，用于表示网格中的点之间的连接关系。D 是归一化的度矩阵，表示每个点的连接数。
    sparse.save_npz('/mnt/mydisk6/lcx666/mstoatd4/mstoatd-main/data/porto/adj.npz', A)
    sparse.save_npz('/mnt/mydisk6/lcx666/mstoatd4/mstoatd-main/data/porto/d_norm.npz', D)

    print('Grid size:', (lat_grid_num, lng_grid_num))
    print('----------Preprocessing----------')
    main()
