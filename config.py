import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=64)

parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--n_cluster', type=int, default=20)

parser.add_argument('--pretrain_lr_s', type=float, default=2e-3)
parser.add_argument('--pretrain_lr_t', type=float, default=2e-3)

parser.add_argument('--lr_s', type=float, default=1e-5)
parser.add_argument('--lr_t', type=float, default=8e-5)

parser.add_argument('--epochs', type=int, default=8)
parser.add_argument('--pretrain_epochs', type=int, default=4)

parser.add_argument("--ratio", type=float, default=0.05, help="ratio of outliers")
parser.add_argument("--distance", type=int, default=3)
parser.add_argument("--fraction", type=float, default=0.1)
parser.add_argument("--obeserved_ratio", type=float, default=1.0)

parser.add_argument("--device", type=str, default='cuda:1')
parser.add_argument("--dataset", type=str, default='porto')
parser.add_argument("--update_mode", type=str, default='rank')

parser.add_argument("--train_num", type=int, default=80000)  # 80000 200000

parser.add_argument("--s1_size", type=int, default=2)
parser.add_argument("--s2_size", type=int, default=4)
#own:
parser.add_argument("--task", type=str, default='test')

parser.add_argument("--mask_token", type=int, default='6069')#s_token_size porto6069 cd 25718 
parser.add_argument("--mask_length", type=int, default=2)
parser.add_argument("--mask_token_x", type=int, default=-1)
parser.add_argument("--mask_token_y", type=int, default=-1)
parser.add_argument('--mask_prob', type=float, default=0.3)
parser.add_argument("--longest", type=int, default=120, help="maximum trajectory length")
parser.add_argument("--grid_embed_size", type=int, default=512)
parser.add_argument("--emb_size", type=int, default=512)
parser.add_argument("--hidden_size_grid", type=int, default=512)
parser.add_argument("--drop_route_rate", type=float, default=0.1)
parser.add_argument("--gps_embed_size", type=int, default=256)
parser.add_argument("--gps_embed_hidden_size", type=int, default=64)
parser.add_argument("--hidden_size_cl", type=int, default=128)
args = parser.parse_args()




