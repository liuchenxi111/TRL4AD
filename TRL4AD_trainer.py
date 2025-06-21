import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from logging_set import get_logger
from TRL4AD import TRL4AD
from utils import auc_score, make_mask, make_len_mask
from cl_loss import get_traj_cl_loss, get_road_cl_loss, get_traj_cluster_loss, get_traj_match_loss


def collate_fn(batch):
    max_len = max(len(x) for x in batch)
    seq_lengths = list(map(len, batch))
    batch_trajs = [x + [[0, [0] * 6]] * (max_len - len(x)) for x in batch]
    return torch.LongTensor(np.array(batch_trajs, dtype=object)[:, :, 0].tolist()), \
        torch.Tensor(np.array(batch_trajs, dtype=object)[:, :, 1].tolist()), np.array(seq_lengths)


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MyDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        data_seqs = self.seqs[index]
        return data_seqs
class MyDataset_gps(Dataset):
    def __init__(self, trajs, trajs_gps):
        assert len(trajs) == len(trajs_gps), "train_trajs and train_trajs_gps must have the same length"
        self.trajs = trajs
        self.trajs_gps = trajs_gps

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, index):
        traj_grid = [t[0] for t in self.trajs[index]]  # Extract only trajectory indices
        traj_gps = [g[0] for g in self.trajs_gps[index]]  # Extract only GPS coordinates
        times = [t[1] for t in self.trajs[index]]  # Extract time information separately
        lengths = len(traj_grid)
        return traj_grid, traj_gps, times, lengths

# def collate_fn_gps(batch):
#     trajs_grid, trajs_gps, times, lengths = zip(*batch)
#
#     max_len = max(lengths)
#
#     # Ensure padding is consistent with trajs_grid and trajs_gps format
#     batch_trajs = [list(x) + [0] * (max_len - len(x)) for x in trajs_grid]
#     batch_gps = [list(x) + [(0, 0)] * (max_len - len(x)) for x in trajs_gps]
#     # ✅ 处理 times，使其成为 Tensor 并填充到 max_len
#     padded_times = [list(x) + [[0, 0, 0, 0, 0, 0]] * (max_len - len(x)) for x in times]
#     times_tensor = torch.tensor(padded_times, dtype=torch.float32)
#     lengths = torch.tensor(lengths, dtype=torch.long)
#     return batch_trajs, batch_gps, times_tensor, lengths
def collate_fn_gps(batch):
    trajs_grid, trajs_gps, times, lengths = zip(*batch)
    max_len = max(lengths)

    # Padding
    batch_trajs = [list(x) + [0] * (max_len - len(x)) for x in trajs_grid]
    batch_gps = [list(x) + [(0, 0)] * (max_len - len(x)) for x in trajs_gps]
    padded_times = [list(x) + [[0] * 6] * (max_len - len(x)) for x in times]

    # Convert to tensor
    gps_tensor = torch.tensor(batch_gps, dtype=torch.float32)

    # 对每条轨迹的x和y进行独立归一化（排除填充值0）
    for i in range(gps_tensor.shape[0]):
        # X坐标归一化
        x = gps_tensor[i, :, 0]
        valid_x = x[x != 0]
        if len(valid_x) > 0:
            x_min, x_max = valid_x.min(), valid_x.max()
            gps_tensor[i, :, 0] = torch.where(
                x != 0,
                (x - x_min) / (x_max - x_min + 1e-8),
                x
            )

        # Y坐标归一化
        y = gps_tensor[i, :, 1]
        valid_y = y[y != 0]
        if len(valid_y) > 0:
            y_min, y_max = valid_y.min(), valid_y.max()
            gps_tensor[i, :, 1] = torch.where(
                y != 0,
                (y - y_min) / (y_max - y_min + 1e-8),
                y
            )

    # 处理其他数据
    times_tensor = torch.tensor(padded_times, dtype=torch.float32)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return batch_trajs, gps_tensor, times_tensor, lengths


def time_convert(times, time_interval):
    return torch.Tensor((times[:, :, 2] + times[:, :, 1] * 60 + times[:, :, 0] * 3600) // time_interval).long()


def savecheckpoint(state, file_name):
    torch.save(state, file_name)


class train_TRL4AD:
    def __init__(self, s_token_size, t_token_size, labels, train_loader, outliers_loader, args):
        self.TRL4AD_S = TRL4AD(s_token_size, s_token_size, args).to(args.device)
        self.TRL4AD_T = TRL4AD(s_token_size, t_token_size, args).to(args.device)
        self.device = args.device  
        self.dataset = args.dataset
        self.n_cluster = args.n_cluster  
        self.hidden_size = args.hidden_size 

        self.crit = nn.CrossEntropyLoss()  
        self.detec = nn.CrossEntropyLoss(reduction='none')  
        self.pretrain_optimizer_s = optim.AdamW([
            {'params': self.TRL4AD_S.parameters()},
        ], lr=args.pretrain_lr_s) 
        self.pretrain_optimizer_t = optim.AdamW([
            {'params': self.TRL4AD_T.parameters()},
        ], lr=args.pretrain_lr_t)
        self.optimizer_s = optim.AdamW([
            {'params': self.TRL4AD_S.parameters()},
        ], lr=args.lr_s)
        self.optimizer_t = optim.Adam([
            {'params': self.TRL4AD_T.parameters()},
        ], lr=args.lr_t)
        self.lr_pretrain_s = StepLR(self.pretrain_optimizer_s, step_size=2, gamma=0.9)
        self.lr_pretrain_t = StepLR(self.pretrain_optimizer_t, step_size=2, gamma=0.9)
        self.train_loader = train_loader
        self.outliers_loader = outliers_loader
        self.pretrained_path = 'models/pretrain_mstoatd_{}.pth'.format(
            args.dataset)
        self.path_checkpoint = 'models/mstoatd_{}.pth'.format(args.dataset)
        self.gmm_path = "models/gmm_{}.pt".format(args.dataset)
        self.gmm_update_path = "models/gmm_update_{}.pt".format(args.dataset)
        self.logger = get_logger("logs/{}.log".format(args.dataset))
        self.labels = labels
        if args.dataset == 'cd':
            self.time_interval = 10
        else:
            self.time_interval = 15
        self.mode = 'train' 
        self.s1_size = args.s1_size
        self.s2_size = args.s2_size
        self.mask_token = args.mask_token
        self.mask_length = args.mask_length
        self.mask_prob = args.mask_prob
        self.mask_token_x = args.mask_token_x
        self.mask_token_y = args.mask_token_y
    def random_mask(self, trajs_grid, trajs_gps, seq_lengths,
                    mask_token, mask_token_x, mask_token_y,
                    mask_length, mask_prob):
        """
        对 grid 和 gps 同步进行随机掩码，适配已有 pretrain() 调用方式。

        参数:
            trajs_grid: Tensor, shape (B, L) — 网格编号轨迹
            trajs_gps: Tensor, shape (B, L, 2) — GPS 坐标轨迹
            seq_lengths: Tensor, shape (B,) — 每条轨迹的真实长度
            mask_token: int — grid 掩码 token
            mask_token_x/y: float — gps 掩码 token
            mask_length: int — 连续 mask 的长度
            mask_prob: float — mask 的概率

        返回:
            masked_grid, masked_gps
        """

        batch_size, max_len = trajs_grid.shape
        device = trajs_grid.device

        masked_grid = trajs_grid.clone()
        masked_gps = trajs_gps.clone()

        for i in range(batch_size):
            actual_len = seq_lengths[i].item()
            if actual_len < mask_length:
                continue  # 长度不足，跳过

            # 可以选的 mask 起点数量
            valid_start = actual_len - mask_length + 1
            num_masks = max(1, int((actual_len / mask_length) * mask_prob))

            # 随机选择 mask 起点
            perm = torch.randperm(valid_start)[:num_masks]
            for start in perm:
                end = start + mask_length
                masked_grid[i, start:end] = mask_token
                masked_gps[i, start:end] = torch.tensor([mask_token_x, mask_token_y], dtype=torch.float32,
                                                        device=device)
        return masked_grid, masked_gps
    def get_masked_label_and_rep(self, tokens, reps, seq_lengths):
        y_label = []
        rep_list = []
        mat2flatten = {}
        now_idx = 0
        for i, length in enumerate(seq_lengths):
            y_label.append(tokens[i, :length])
            rep_list.append(reps[i, :length])
            for l in range(length):
                mat2flatten[(i, l)] = now_idx
                now_idx += 1
        y_label = torch.cat(y_label, dim=0).to(tokens.device)
        flat_rep = torch.cat(rep_list, dim=0)

        return y_label, flat_rep, mat2flatten
    
    def get_hidden(self):
        checkpoint = torch.load(self.path_checkpoint)
        self.TRL4AD_S.load_state_dict(checkpoint['model_state_dict_s'])
        self.TRL4AD_S.eval()
        with torch.no_grad():
            z = []
            for batch in self.train_loader:
                trajs, trajs_gps, times, seq_lengths = batch
                trajs = torch.tensor(trajs, dtype=torch.long).to(self.device)
                trajs_gps = trajs_gps.to(self.device)
                batch_size = len(trajs)
                _, _, _, hidden, _, _, _, _ = self.TRL4AD_S(trajs, trajs, trajs_gps, trajs_gps, times, seq_lengths,
                                                              batch_size, "pretrain", -1, self.mask_token,
                                                              self.mask_token_x, self.mask_token_y)
                z.append(hidden.squeeze(0))
            z = torch.cat(z, dim=0)
        return z

    def train_gmm(self):
        self.TRL4AD_S.eval()  
        self.TRL4AD_T.eval()
        checkpoint = torch.load(self.pretrained_path)  
        self.TRL4AD_S.load_state_dict(checkpoint['model_state_dict_s'])
        self.TRL4AD_T.load_state_dict(checkpoint['model_state_dict_t'])

        with torch.no_grad():
            z_s = []  
            z_t = []
            for batch_idx, batch in enumerate(self.train_loader):
                trajs, trajs_gps, times, seq_lengths = batch
                trajs = torch.tensor(trajs, dtype=torch.long).to(self.device)
                trajs_gps = trajs_gps.to(self.device)
                batch_size = len(trajs)
                mask = make_mask(make_len_mask(trajs)).to(self.device)  
                self.pretrain_optimizer_s.zero_grad()  
                self.pretrain_optimizer_t.zero_grad()
                _, _, _, hidden_s, grid_unpooled, grid_pooled, gps_unpooled, gps_pooled = self.TRL4AD_S(trajs, trajs,trajs_gps,
                                                                                                          trajs_gps, times,
                                                                                                          seq_lengths,
                                                                                                          batch_size,
                                                                                                          "pretrain",
                                                                                                          -1,
                                                                                                          self.mask_token,
                                                                                                          self.mask_token_x,
                                                                                                          self.mask_token_y) 
                _, _, _, hidden_t, _, _, _, _ = self.TRL4AD_T(trajs, trajs, trajs_gps, trajs_gps, times,
                                                                seq_lengths, batch_size, "pretrain", -1,
                                                                self.mask_token, self.mask_token_x, self.mask_token_y)
                z_s.append(hidden_s.squeeze(0))
                z_t.append(hidden_t.squeeze(0))
            z_s = torch.cat(z_s, dim=0)
            z_t = torch.cat(z_t, dim=0)  

        self.logger.info('Start fitting gaussian mixture model!')  

        self.gmm_s = GaussianMixture(n_components=self.n_cluster, covariance_type="diag",
                                     n_init=1)
        self.gmm_s.fit(z_s.cpu().numpy())

        self.gmm_t = GaussianMixture(n_components=self.n_cluster, covariance_type="diag",
                                     n_init=1)  
        self.gmm_t.fit(z_t.cpu().numpy())

    def save_weights_for_MSTOATD(self):
        savecheckpoint({"gmm_s_mu_prior": self.gmm_s.means_,
                        "gmm_s_pi_prior": self.gmm_s.weights_,
                        "gmm_s_logvar_prior": self.gmm_s.covariances_,
                        "gmm_t_mu_prior": self.gmm_t.means_,
                        "gmm_t_pi_prior": self.gmm_t.weights_,
                        "gmms_t_logvar_prior": self.gmm_t.covariances_}, self.gmm_path)

    def train_gmm_update(self):

        checkpoint = torch.load(self.path_checkpoint)
        self.TRL4AD_S.load_state_dict(checkpoint['model_state_dict_s'])
        self.TRL4AD_S.eval()

        with torch.no_grad():
            z = []
            for batch in self.train_loader:
                trajs, trajs_gps, times, seq_lengths = batch
                trajs = torch.tensor(trajs, dtype=torch.long).to(self.device)
                trajs_gps = trajs_gps.to(self.device)
                batch_size = len(trajs)
                _, _, _, hidden, _, _, _, _ = self.TRL4AD_S(trajs, trajs,trajs_gps,trajs_gps,times, seq_lengths,
                                                batch_size,"pretrain",-1,self.mask_token,self.mask_token_x,self.mask_token_y) 
                z.append(hidden.squeeze(0))

            z = torch.cat(z, dim=0)
        self.logger.info('Start fitting gaussian mixture model!')
        self.gmm = GaussianMixture(n_components=self.n_cluster, covariance_type="diag", n_init=3)
        self.gmm.fit(z.cpu().numpy())
        savecheckpoint({"gmm_update_weights": self.gmm.weights_,
                        "gmm_update_means": self.gmm.means_,
                        "gmm_update_covariances": self.gmm.covariances_,
                        "gmm_update_precisions_cholesky": self.gmm.precisions_cholesky_}, self.gmm_update_path)


    def pretrain(self, epoch):
        self.TRL4AD_S.train() 
        self.TRL4AD_T.train()
        epo_loss = 0  
        total_loss = 0
        total_s_loss = 0
        total_t_loss = 0
        total_grid_loss = 0
        total_gps_loss = 0
        total_match_loss = 0
        total_batches = len(self.train_loader)  
        for batch_idx, batch in enumerate(self.train_loader):
            trajs, trajs_gps, times, seq_lengths = batch
            trajs = torch.tensor(trajs, dtype=torch.long).to(self.device)
            trajs_gps = trajs_gps.to(self.device)
            batch_size = len(trajs) 
            masked_grid, masked_gps = self.random_mask(trajs, trajs_gps,seq_lengths,self.mask_token,self.mask_token_x, self.mask_token_y, self.mask_length, self.mask_prob)
            masked_grid = masked_grid.to(self.device)
            masked_gps = masked_gps.to(self.device)
            mask = make_mask(make_len_mask(trajs)).to(self.device) 

            self.pretrain_optimizer_s.zero_grad()  
            self.pretrain_optimizer_t.zero_grad()
            output_s, _, _, _, grid_unpooled, grid_pooled ,gps_unpooled, gps_pooled = self.TRL4AD_S(trajs,masked_grid,trajs_gps, masked_gps, times, seq_lengths, batch_size, "pretrain",-1,self.mask_token,self.mask_token_x, self.mask_token_y)  # 前向传播(300,49,6069)
            output_t, _, _, _,_, _, _, _ = self.TRL4AD_T(trajs,masked_grid,trajs_gps, masked_gps, times, seq_lengths, batch_size, "pretrain", -1,self.mask_token,self.mask_token_x, self.mask_token_y)
            ###########对比损失
            gps_pooled_cl = self.TRL4AD_S.gps_proj_head(gps_pooled)
            grid_pooled_cl = self.TRL4AD_S.route_proj_head(grid_pooled)
            match_loss = get_traj_match_loss(self, gps_pooled_cl, grid_pooled_cl, self.TRL4AD_S, len(trajs), tau=0.07)
            #######掩码损失
            times = time_convert(times, self.time_interval) 
            y_grid_label, grid_rep, mat2flatten = self.get_masked_label_and_rep(trajs, grid_unpooled,
                                                                                seq_lengths)
            y_gps_label, gps_rep, _ = self.get_masked_label_and_rep(trajs_gps, gps_unpooled, seq_lengths)
            y_gps_label = y_gps_label.to(torch.float32) 
            masked_pos = torch.nonzero(trajs != masked_grid, as_tuple=False)
            masked_pos = [mat2flatten[tuple(pos.tolist())] for pos in masked_pos]
            grid_preds = self.TRL4AD_S.route_mlm_head(grid_rep)[masked_pos].to(self.device)
            gps_preds = self.TRL4AD_S.gps_mlm_head(gps_rep)[masked_pos].to(self.device)
            y_grid_masked = y_grid_label[masked_pos].to(self.device)
            y_grid_masked = y_grid_masked.long()  
            y_gps_masked = y_gps_label[masked_pos]
            grid_mlm_loss = nn.CrossEntropyLoss()(grid_preds, y_grid_masked)
            gps_mse_loss = (nn.MSELoss()(gps_preds, y_gps_masked))
            loss_s = self.crit(output_s[mask == 1], trajs.to(self.device)[mask == 1])  
            loss_t = self.crit(output_t[mask == 1], times.to(self.device)[mask == 1])
            loss = loss_t+loss_s + grid_mlm_loss + gps_mse_loss + match_loss
            loss.backward()  

            self.pretrain_optimizer_s.step()  
            self.pretrain_optimizer_t.step()
            epo_loss += loss.item()  
            total_loss += loss.item()
            total_s_loss += loss_s.item()
            total_t_loss += loss_t.item()
            total_grid_loss += grid_mlm_loss.item()
            total_gps_loss += gps_mse_loss.item()
            total_match_loss += match_loss.item()
        self.lr_pretrain_s.step()
        self.lr_pretrain_t.step()

        # **计算 & 打印整个 epoch 的平均损失**
        avg_loss = total_loss / total_batches
        avg_s_loss = total_s_loss / total_batches
        avg_t_loss = total_t_loss / total_batches
        avg_grid_loss = total_grid_loss / total_batches
        avg_gps_loss = total_gps_loss / total_batches
        avg_match_loss = total_match_loss / total_batches

        print(f"✅ Epoch [{epoch + 1}] completed | Avg Total Loss: {avg_loss:.4f} ")

        self.logger.info(
                f"Epoch {epoch + 1} | Avg Total Loss: {avg_loss:.4f}")
        checkpoint = {"model_state_dict_s": self.TRL4AD_S.state_dict(),
                      "model_state_dict_t": self.TRL4AD_T.state_dict()}  
        torch.save(checkpoint, self.pretrained_path)  
    

            
    def train(self, epoch, total_reconstruction_loss=None):
        self.TRL4AD_S.train()
        self.TRL4AD_T.train()
        epo_loss = 0  
        total_loss = 0
        total_s_loss = 0
        total_t_loss = 0
        total_grid_loss = 0
        total_gps_loss = 0
        total_match_loss = 0
        total_reconstruction_loss_s  = 0
        total_reconstruction_loss_t = 0
        total_gaussian_loss_s  = 0
        total_gaussian_loss_t = 0
        total_category_loss_s  = 0
        total_category_loss_t  = 0
        total_batches = len(self.train_loader) 
        for batch_idx, batch in enumerate(self.train_loader): 
            trajs, trajs_gps, times, seq_lengths = batch
            trajs = torch.tensor(trajs, dtype=torch.long).to(self.device)
            trajs_gps = trajs_gps.to(self.device)
            batch_size = len(trajs)  
            mask = make_mask(make_len_mask(trajs)).to(self.device) 
            self.pretrain_optimizer_s.zero_grad() 
            self.pretrain_optimizer_t.zero_grad()
            x_hat_s, mu_s, log_var_s, z_s, grid_unpooled, grid_pooled, gps_unpooled, gps_pooled = self.TRL4AD_S(trajs,
                                                                                                      trajs,
                                                                                                      trajs_gps,
                                                                                                      trajs_gps, times,
                                                                                                      seq_lengths,
                                                                                                      batch_size,
                                                                                                      "train", -1,
                                                                                                      self.mask_token,
                                                                                                      self.mask_token_x,
                                                                                                      self.mask_token_y)  
            x_hat_t, mu_t, log_var_t, z_t, _, _, _, _ = self.TRL4AD_T(trajs, trajs, trajs_gps, trajs_gps, times,
                                                            seq_lengths, batch_size, "train", -1, self.mask_token,
                                                            self.mask_token_x, self.mask_token_y)
            ###########对比损失
            gps_pooled_cl = self.TRL4AD_S.gps_proj_head(gps_pooled)
            grid_pooled_cl = self.TRL4AD_S.route_proj_head(grid_pooled)
            match_loss = get_traj_match_loss(self, gps_pooled_cl, grid_pooled_cl, self.TRL4AD_S, len(trajs), tau=0.07)
            #######掩码损失
            times = time_convert(times, self.time_interval)  
            y_grid_label, grid_rep, mat2flatten = self.get_masked_label_and_rep(trajs, grid_unpooled,
                                                                                seq_lengths)
            y_gps_label, gps_rep, _ = self.get_masked_label_and_rep(trajs_gps, gps_unpooled, seq_lengths)
            y_gps_label = y_gps_label.to(torch.float32) 
            grid_preds = self.TRL4AD_S.route_mlm_head(grid_rep).to(self.device)
            gps_preds = self.TRL4AD_S.gps_mlm_head(gps_rep).to(self.device)
            y_grid_masked = y_grid_label.to(self.device)
            y_grid_masked = y_grid_masked.long()  # 转换为整数类型
            y_gps_masked = y_gps_label
            loss_grid = nn.CrossEntropyLoss()(grid_preds, y_grid_masked)
            gps_mse_loss = (nn.MSELoss()(gps_preds, y_gps_masked))
            ##########
            loss_s,reconstruction_loss_s,gaussian_loss_s,category_loss_s = self.Loss(x_hat_s, trajs.to(self.device), mu_s.squeeze(0), log_var_s.squeeze(0),
                               z_s.squeeze(0), 's', mask)
            loss_t,reconstruction_loss_t,gaussian_loss_t,category_loss_t = self.Loss(x_hat_t, times.to(self.device), mu_t.squeeze(0), log_var_t.squeeze(0),
                               z_t.squeeze(0), 't', mask)
            loss = loss_t + loss_s + loss_grid + gps_mse_loss + match_loss
            loss.backward()  # 反向传播以计算梯度。

            self.pretrain_optimizer_s.step()  # 更新优化器以调整模型参数。
            self.pretrain_optimizer_t.step()
            epo_loss += loss.item()  # 累加当前批次的损失到总损失中。
            total_loss += loss.item()
            total_s_loss += loss_s.item()
            total_t_loss += loss_t.item()
            total_grid_loss += loss_grid.item()
            total_gps_loss += gps_mse_loss.item()
            total_match_loss += match_loss.item()
            total_reconstruction_loss_s += reconstruction_loss_s.item()
            total_reconstruction_loss_t += reconstruction_loss_t.item()
            total_gaussian_loss_s += gaussian_loss_s.item()
            total_gaussian_loss_t += gaussian_loss_t.item()
            total_category_loss_s += category_loss_s.item()
            total_category_loss_t += category_loss_t.item()
        self.lr_pretrain_s.step()
        self.lr_pretrain_t.step()

        avg_loss = total_loss / total_batches
        avg_s_loss = total_s_loss / total_batches
        avg_t_loss = total_t_loss / total_batches
        avg_grid_loss = total_grid_loss / total_batches
        avg_gps_loss = total_gps_loss / total_batches
        avg_match_loss = total_match_loss / total_batches
        avg_reconstruction_loss_s = total_reconstruction_loss_s/ total_batches
        avg_reconstruction_loss_t = total_reconstruction_loss_t/ total_batches
        avg_gaussian_loss_s = total_gaussian_loss_s/ total_batches
        avg_gaussian_loss_t = total_gaussian_loss_t/ total_batches
        avg_category_loss_s = total_category_loss_s/ total_batches
        avg_category_loss_t = total_category_loss_t/ total_batches

        
        self.logger.info(
            f"Epoch {epoch + 1} | Avg Total Loss: {avg_loss:.4f} ")
        checkpoint = {"model_state_dict_s": self.TRL4AD_S.state_dict(),
                      "model_state_dict_t": self.TRL4AD_T.state_dict()}  # 创建一个字典，保存两个模型的状态字典。
        torch.save(checkpoint, self.path_checkpoint)  # 将模型的状态保存到指定路径。
    


    def detection(self):
        self.TRL4AD_S.eval()
        all_likelihood_s = []
        self.TRL4AD_T.eval()
        all_likelihood_t = []

        with torch.no_grad():

            for batch in self.outliers_loader:
                trajs,trajs_gps, times, seq_lengths = batch
                trajs = torch.tensor(trajs, dtype=torch.long).to(self.device)
                trajs_gps = trajs_gps.to(self.device)
                batch_size = len(trajs)
                mask = make_mask(make_len_mask(trajs)).to(self.device)
                times_token = time_convert(times, self.time_interval)

                c_likelihood_s = []
                c_likelihood_t = []

                for c in range(self.n_cluster): 
                    output_s, _, _, _, _, _, _, _ = self.TRL4AD_S(trajs, trajs, trajs_gps, trajs_gps, times,
                                                                    seq_lengths, batch_size, "test", c,
                                                                    self.mask_token, self.mask_token_x,
                                                                    self.mask_token_y)  # 前向传播(300,49,6069)
                    output_t, _, _, _, _, _, _, _ = self.TRL4AD_T(trajs, trajs, trajs_gps, trajs_gps, times,
                                                                    seq_lengths, batch_size, "test", c,
                                                                    self.mask_token, self.mask_token_x,
                                                                    self.mask_token_y)
                    likelihood_s = - self.detec(output_s.reshape(-1, output_s.shape[-1]),
                                                trajs.to(self.device).reshape(-1))
                    likelihood_s = torch.exp(
                        torch.sum(mask * (likelihood_s.reshape(batch_size, -1)), dim=-1) / torch.sum(mask, 1))
                    likelihood_t = - self.detec(output_t.reshape(-1, output_t.shape[-1]),
                                                times_token.to(self.device).reshape(-1))
                    likelihood_t = torch.exp(
                        torch.sum(mask * (likelihood_t.reshape(batch_size, -1)), dim=-1) / torch.sum(mask, 1))

                    c_likelihood_s.append(likelihood_s.unsqueeze(0))
                    c_likelihood_t.append(likelihood_t.unsqueeze(0))

                all_likelihood_s.append(torch.cat(c_likelihood_s).max(0)[0])
                all_likelihood_t.append(torch.cat(c_likelihood_t).max(0)[0])

        likelihood_s = torch.cat(all_likelihood_s, dim=0)
        likelihood_t = torch.cat(all_likelihood_t, dim=0)
        pr_auc = auc_score(self.labels, (1 - likelihood_s * likelihood_t).cpu().detach().numpy())
        return pr_auc

    def gaussian_pdf_log(self, x, mu, log_var):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_var + (x - mu).pow(2) / torch.exp(log_var), 1))

    def gaussian_pdfs_log(self, x, mus, log_vars):
        G = []
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_vars[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    def Loss(self, x_hat, targets, z_mu, z_sigma2_log, z, mode, mask):
        if mode == 's':
            pi = self.TRL4AD_S.pi_prior
            log_sigma2_c = self.TRL4AD_S.log_var_prior
            mu_c = self.TRL4AD_S.mu_prior
        elif mode == 't':
            pi = self.TRL4AD_T.pi_prior
            log_sigma2_c = self.TRL4AD_T.log_var_prior
            mu_c = self.TRL4AD_T.mu_prior
    
        reconstruction_loss = self.crit(x_hat[mask == 1], targets[mask == 1])
        gaussian_loss = torch.mean(torch.mean(self.gaussian_pdf_log(z, z_mu, z_sigma2_log).unsqueeze(1) -
                                              self.gaussian_pdfs_log(z, mu_c, log_sigma2_c), dim=1), dim=-1).mean()

        z = z.unsqueeze(1)
        mu_c = mu_c.unsqueeze(0)
        log_sigma2_c = log_sigma2_c.unsqueeze(0)

        logits = - torch.sum(torch.pow(z - mu_c, 2) / torch.exp(log_sigma2_c), dim=-1)
        logits = F.softmax(logits, dim=-1) + 1e-10

        logits_mean = torch.mean(logits, dim=0)
        category_loss = torch.mean(pi * torch.log(logits_mean))
        if torch.isnan(gaussian_loss) or torch.isnan(category_loss) or torch.isnan(reconstruction_loss):
            print("[❗] Loss contains NaN!")
        loss = reconstruction_loss + gaussian_loss / self.hidden_size + category_loss * 0.001
        return loss,reconstruction_loss,gaussian_loss / self.hidden_size,category_loss

    def load_TRL4AD(self):
        checkpoint = torch.load(self.pretrained_path)
        self.TRL4AD_S.load_state_dict(checkpoint['model_state_dict_s'])
        self.TRL4AD_T.load_state_dict(checkpoint['model_state_dict_t'])

        gmm_params = torch.load(self.gmm_path)

        self.TRL4AD_S.pi_prior.data = torch.from_numpy(gmm_params['gmm_s_pi_prior']).to(self.device)
        self.TRL4AD_S.mu_prior.data = torch.from_numpy(gmm_params['gmm_s_mu_prior']).to(self.device)
        self.TRL4AD_S.log_var_prior.data = torch.from_numpy(gmm_params['gmm_s_logvar_prior']).to(self.device)

        self.TRL4AD_T.pi_prior.data = torch.from_numpy(gmm_params['gmm_t_pi_prior']).to(self.device)
        self.TRL4AD_T.mu_prior.data = torch.from_numpy(gmm_params['gmm_t_mu_prior']).to(self.device)
        self.TRL4AD_T.log_var_prior.data = torch.from_numpy(gmm_params['gmms_t_logvar_prior']).to(self.device)

    def get_prob(self, z):
        gmm = GaussianMixture(n_components=self.n_cluster, covariance_type='diag')
        gmm_params = torch.load(self.gmm_update_path)
        gmm.precisions_cholesky_ = gmm_params['gmm_update_precisions_cholesky']
        gmm.weights_ = gmm_params['gmm_update_weights']
        gmm.means_ = gmm_params['gmm_update_means']
        gmm.covariances_ = gmm_params['gmm_update_covariances']

        probs = gmm.predict_proba(z)

        for label in range(self.n_cluster):
            np.save('/mnt/mydisk6/lcx666/mstoatd6/mstoatd-main/probs/probs_{}_{}.npy'.format(label, self.dataset),
                    np.sort(-probs[:, label]))
