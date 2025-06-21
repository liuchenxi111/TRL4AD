import os
from functools import partial
from multiprocessing import Pool, Manager

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.model_selection import train_test_split

from preprocess_utils import in_boundary, cutting_trajs, convert_date, timestamp_gap, grid_mapping, generate_matrix

# ---- 预设函数：根据你已有环境中实现的功能 ----
# 你应替换这些为你已有的定义：
# grid_mapping, generate_matrix, timestamp_gap, in_boundary, convert_date

def cutting_trajs(traj_seq, traj_gps_seq, longest, shortest):
    cut_grid = []
    cut_gps = []
    total_len = len(traj_seq)
    i = 0
    while i + longest <= total_len:
        cut_grid.append(traj_seq[i:i + longest])
        cut_gps.append(traj_gps_seq[i:i + longest])
        i += longest
    # 剩余部分如果够长也保留
    if total_len - i >= shortest:
        cut_grid.append(traj_seq[i:])
        cut_gps.append(traj_gps_seq[i:])
    return cut_grid, cut_gps


def preprocess(file, traj_path, shortest, longest, boundary, lat_size, lng_size, lng_grid_num,
               convert_date, timestamp_gap, in_boundary, cutting_trajs,
               traj_nums, point_nums):
    np.random.seed(1234)
    data = pd.read_csv(f"{traj_path}/{file}", header=None)
    data.columns = ['id', 'lat', 'lon', 'state', 'timestamp']
    data = data.sort_values(by=['id', 'timestamp'])
    data = data[data['state'] == 1]

    trajs_grid = []
    trajs_gps = []
    traj_seq = []
    traj_gps_seq = []
    valid = True

    pre_point = data.iloc[0]

    for point in data.itertuples():
        if point.id == pre_point.id and timestamp_gap(pre_point.timestamp, point.timestamp) <= 20:
            if in_boundary(point.lat, point.lon, boundary):
                grid_i = int((point.lat - boundary['min_lat']) / lat_size)
                grid_j = int((point.lon - boundary['min_lng']) / lng_size)
                grid_id = grid_i * lng_grid_num + grid_j

                traj_seq.append([grid_id, convert_date(point.timestamp)])
                traj_gps_seq.append([(point.lat, point.lon), convert_date(point.timestamp)])
            else:
                valid = False
        else:
            if valid:
                if shortest <= len(traj_seq) <= longest:
                    trajs_grid.append(traj_seq)
                    trajs_gps.append(traj_gps_seq)
                elif len(traj_seq) > longest:
                    cut_grid, cut_gps = cutting_trajs(traj_seq, traj_gps_seq, longest, shortest)
                    trajs_grid += cut_grid
                    trajs_gps += cut_gps

            traj_seq = []
            traj_gps_seq = []
            valid = True
        pre_point = point

    traj_nums.append(len(trajs_grid))
    point_nums.append(sum([len(traj) for traj in trajs_grid]))

    train_grid, test_grid = train_test_split(trajs_grid, test_size=0.2, random_state=42)
    train_gps, test_gps = train_test_split(trajs_gps, test_size=0.2, random_state=42)

    base_name = file[:8]
    np.save(f"{traj_path}/train_grid_{base_name}.npy", np.array(train_grid, dtype=object))
    np.save(f"{traj_path}/test_grid_{base_name}.npy", np.array(test_grid, dtype=object))
    np.save(f"{traj_path}/train_gps_{base_name}.npy", np.array(train_gps, dtype=object))
    np.save(f"{traj_path}/test_gps_{base_name}.npy", np.array(test_gps, dtype=object))


def batch_preprocess(path_list):
    manager = Manager()
    traj_nums = manager.list()
    point_nums = manager.list()
    pool = Pool(processes=10)
    pool.map(partial(preprocess, traj_path=traj_path, shortest=shortest, longest=longest, boundary=boundary,
                     lat_size=lat_size, lng_size=lng_size, lng_grid_num=lng_grid_num, convert_date=convert_date,
                     timestamp_gap=timestamp_gap, in_boundary=in_boundary, cutting_trajs=cutting_trajs,
                     traj_nums=traj_nums, point_nums=point_nums), path_list)
    pool.close()
    pool.join()

    print("Total trajectory num:", sum(traj_nums))
    print("Total point num:", sum(point_nums))


def merge(path_list):
    res_train_grid, res_test_grid = [], []
    res_train_gps, res_test_gps = [], []

    for file in path_list:
        base_name = file[:8]
        train_grid = np.load(f"{traj_path}/train_grid_{base_name}.npy", allow_pickle=True)
        test_grid = np.load(f"{traj_path}/test_grid_{base_name}.npy", allow_pickle=True)
        train_gps = np.load(f"{traj_path}/train_gps_{base_name}.npy", allow_pickle=True)
        test_gps = np.load(f"{traj_path}/test_gps_{base_name}.npy", allow_pickle=True)

        res_train_grid.append(train_grid)
        res_test_grid.append(test_grid)
        res_train_gps.append(train_gps)
        res_test_gps.append(test_gps)

    return (np.concatenate(res_train_grid, axis=0),
            np.concatenate(res_test_grid, axis=0),
            np.concatenate(res_train_gps, axis=0),
            np.concatenate(res_test_gps, axis=0))


def main():
    path_list = os.listdir(traj_path)
    path_list.sort(key=lambda x: x.split('.'))
    path_list = path_list[:10]

    batch_preprocess(path_list)
    train_grid, test_grid, train_gps, test_gps = merge(path_list[:3])

    np.save(f"{traj_path}/train_grid_init.npy", train_grid)
    np.save(f"{traj_path}/test_grid_init.npy", test_grid)
    np.save(f"{traj_path}/train_gps_init.npy", train_gps)
    np.save(f"{traj_path}/test_gps_init.npy", test_gps)

    print('Finished!')


if __name__ == "__main__":
    traj_path = "/mnt/mydisk6/lcx666/mstoatd4/mstoatd-main/data/cd"
    grid_size = 0.1
    shortest, longest = 30, 100
    boundary = {'min_lat': 30.6, 'max_lat': 30.75, 'min_lng': 104, 'max_lng': 104.16}

    lat_size, lng_size, lat_grid_num, lng_grid_num = grid_mapping(boundary, grid_size)
    A, D = generate_matrix(lat_grid_num, lng_grid_num)

    sparse.save_npz(f'{traj_path}/adj.npz', A)
    sparse.save_npz(f'{traj_path}/d_norm.npz', D)

    print('Grid size:', (lat_grid_num, lng_grid_num))
    print('----------Preprocessing----------')
    main()
