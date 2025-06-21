import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import args
from TRL4AD_trainer import train_TRL4AD, MyDataset, seed_torch, collate_fn,MyDataset_gps,collate_fn_gps


def main():
    if args.dataset == 'porto':
        train_trajs = np.load(
            'data/porto/train_trajs_init.npy'.format(args.dataset),
            allow_pickle=True)
        train_trajs_gps = np.load(
            'data/porto/train_data_gps_init_raw.npy'.format(args.dataset),
            allow_pickle=True)
        test_trajs_grid = np.load(
            'data/{}/outliers_data_init_grid_{}_{}_{}.npy'.format(
                args.dataset,
                args.distance,
                args.fraction,
                args.obeserved_ratio),
            allow_pickle=True)
        test_trajs_gps = np.load(
            'data/{}/outliers_data_init_gps_{}_{}_{}.npy'.format(args.dataset,
                                                                                                           args.distance,
                                                                                                           args.fraction,
                                                                                                           args.obeserved_ratio),
            allow_pickle=True)

        # 加载测试数据集，它包含异常值，文件名包含距离、比例和观察比例参数。
        outliers_idx = np.load(
            "data/{}/outliers_idx_init_gps_{}_{}_{}.npy".format(args.dataset,
                                                                                                          args.distance,
                                                                                                          args.fraction,
                                                                                                          args.obeserved_ratio),
            allow_pickle=True)
    else:
        train_trajs = np.load(
            'data/cd/train_grid_init.npy'.format(args.dataset),
            allow_pickle=True)
        train_trajs_gps = np.load(
            'data/cd/train_gps_init.npy'.format(args.dataset),
            allow_pickle=True)

        test_trajs_grid = np.load(
            'data/{}/outliers_data_init_grid_{}_{}_{}.npy'.format(
                args.dataset,
                args.distance,
                args.fraction,
                args.obeserved_ratio),
            allow_pickle=True)
        test_trajs_gps = np.load(
            'data/{}/outliers_data_init_gps_{}_{}_{}.npy'.format(args.dataset,
                                                                                                           args.distance,
                                                                                                           args.fraction,
                                                                                                           args.obeserved_ratio),
            allow_pickle=True)

        outliers_idx = np.load(
            "data/{}/outliers_idx_init_gps_{}_{}_{}.npy".format(args.dataset,
                                                                                                          args.distance,
                                                                                                          args.fraction,
                                                                                                          args.obeserved_ratio),
            allow_pickle=True)

    train_data = MyDataset_gps(train_trajs, train_trajs_gps)
    test_data = MyDataset_gps(test_trajs_grid, test_trajs_gps)
    labels = np.zeros(len(test_trajs_grid))
    for i in outliers_idx:
        labels[i] = 1
    labels = labels

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_gps,
                              num_workers=8, pin_memory=True)
    outliers_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_gps,
                                 num_workers=8, pin_memory=True)
    TRL4AD = train_TRL4AD(s_token_size, t_token_size, labels, train_loader, outliers_loader, args)
    if args.task == 'train':
        TRL4AD.logger.info("Start pretraining!")
        for epoch in range(args.pretrain_epochs):
            TRL4AD.pretrain(epoch)
        TRL4AD.train_gmm()
        TRL4AD.save_weights_for_MSTOATD()
        TRL4AD.logger.info("Start training!")
        TRL4AD.load_TRL4AD()
        for epoch in range(args.epochs):
            TRL4AD.train(epoch)

    if args.task == 'test':
        TRL4AD.logger.info('Start testing!')
        TRL4AD.logger.info("d = {}".format(args.distance) + ", " + chr(945) + " = {}".format(args.fraction) + ", "
              + chr(961) + " = {}".format(args.obeserved_ratio))

        checkpoint = torch.load(TRL4AD.path_checkpoint)
        TRL4AD.TRL4AD_S.load_state_dict(checkpoint['model_state_dict_s'])
        TRL4AD.TRL4AD_T.load_state_dict(checkpoint['model_state_dict_t'])
        pr_auc = TRL4AD.detection()
        pr_auc = "%.4f" % pr_auc
        TRL4AD.logger.info("PR_AUC: {}".format(pr_auc))

    if args.task == 'train':
        TRL4AD.train_gmm_update()
        z = TRL4AD.get_hidden()
        TRL4AD.get_prob(z.cpu())


if __name__ == "__main__":

    if args.dataset == 'porto':
        s_token_size = 51 * 119#grid_size=100
        t_token_size = 5760

    elif args.dataset == 'cd':
        s_token_size = 167 * 154#grid_size=100
        t_token_size = 8640

    main()
