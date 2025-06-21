import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from temporal import TemporalEmbedding



class co_attention(nn.Module):
    def __init__(self, dim):
        super(co_attention, self).__init__()

        self.Wq_s = nn.Linear(dim, dim, bias=False)#in_feature=128,out_feature=128,bias=False
        self.Wk_s = nn.Linear(dim, dim, bias=False)#in_feature=128,out_feature=128,bias=False
        self.Wv_s = nn.Linear(dim, dim, bias=False)#in_feature=128,out_feature=128,bias=False

        self.Wq_t = nn.Linear(dim, dim, bias=False)#in_feature=128,out_feature=128,bias=False
        self.Wk_t = nn.Linear(dim, dim, bias=False)#in_feature=128,out_feature=128,bias=False
        self.Wv_t = nn.Linear(dim, dim, bias=False)#in_feature=128,out_feature=128,bias=False

        self.dim_k = dim ** 0.5

        self.FFN_s = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )

        self.FFN_t = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )

        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, seq_s, seq_t):
        seq_t = seq_t.unsqueeze(2)
        seq_s = seq_s.unsqueeze(2)

        q_s, k_s, v_s = self.Wq_s(seq_t), self.Wk_s(seq_s), self.Wv_s(seq_s)
        q_t, k_t, v_t = self.Wq_t(seq_s), self.Wk_t(seq_t), self.Wv_t(seq_t)

        coatt_s = F.softmax(torch.matmul(q_s / self.dim_k, k_s.transpose(2, 3)), dim=-1)
        coatt_t = F.softmax(torch.matmul(q_t / self.dim_k, k_t.transpose(2, 3)), dim=-1)

        att_s = self.layer_norm(self.FFN_s(torch.matmul(coatt_s, v_s)) + torch.matmul(coatt_s, v_s))
        att_t = self.layer_norm(self.FFN_t(torch.matmul(coatt_t, v_t)) + torch.matmul(coatt_t, v_t))

        return att_s.squeeze(2), att_t.squeeze(2)


class state_attention(nn.Module):#状态注意力机制
    def __init__(self, args):
        super(state_attention, self).__init__()

        self.w_omega = nn.Parameter(torch.Tensor(args.hidden_size, args.hidden_size))#(512*512)用于转换输入序列
        self.u_omega = nn.Parameter(torch.Tensor(args.hidden_size, 1))#(512*1)用于计算注意力分数。
        nn.init.uniform_(self.w_omega, -0.1, 0.1)#使用均匀分布初始化 w_omega 和 u_omega 参数，范围在 -0.1 到 0.1 之间。
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, seq):
        u = torch.tanh(torch.matmul(seq, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        scored_outputs = seq * att_score
        return scored_outputs.sum(1)

class TransformerModel(nn.Module):  # vanilla transformer
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output
##增加残差连接
# class TransformerModel(nn.Module):
#     def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
#         super(TransformerModel, self).__init__()
#         encoder_layers = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
#         # 添加层归一化，提高稳定性
#         self.layer_norm = nn.LayerNorm(input_size)

#     def forward(self, src, src_mask, src_key_padding_mask):
#         # 添加残差连接
#         output = src + self.transformer_encoder(src, src_mask, src_key_padding_mask)
#         # 添加归一化
#         output = self.layer_norm(output)
#         return output
class TRL4AD(nn.Module):
    def __init__(self, token_size, token_size_out, args):
        super(TRL4AD, self).__init__()
        self.emb_size = args.embedding_size
        self.device = args.device
        self.n_cluster = args.n_cluster
        self.dataset = args.dataset
        self.s1_size = args.s1_size
        self.s2_size = args.s2_size
        self.pi_prior = nn.Parameter(torch.ones(args.n_cluster) / args.n_cluster)
        self.mu_prior = nn.Parameter(torch.randn(args.n_cluster, args.hidden_size))
        self.log_var_prior = nn.Parameter(torch.zeros(args.n_cluster, args.hidden_size))
        self.embedding = nn.Embedding(token_size, args.embedding_size)
        self.encoder_s1 = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)
        self.encoder_s2 = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)
        self.encoder_s3 = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)
        self.decoder = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)
        self.fc_mu = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc_logvar = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.fc_out = nn.Linear(args.hidden_size, token_size_out)
        self.nodes = torch.arange(token_size, dtype=torch.long).to(args.device)
        self.adj = sparse.load_npz("data/{}/adj.npz".format(args.dataset))
        self.d_norm = sparse.load_npz("data/{}/d_norm.npz".format(args.dataset))

        if args.dataset == 'porto':
            self.V = nn.Parameter(torch.Tensor(token_size, token_size))#(6069,6069)
        else:
            self.V = nn.Parameter(torch.Tensor(args.embedding_size, args.embedding_size))
        self.W1 = nn.Parameter(torch.ones(1) / 3)#[0.3333]
        self.W2 = nn.Parameter(torch.ones(1) / 3)#[0.3333]
        self.W3 = nn.Parameter(torch.ones(1) / 3)#[0.3333]
        self.co_attention = co_attention(args.embedding_size).to(args.device)
        self.d2v = TemporalEmbedding(args.device)
        self.w_omega = nn.Parameter(torch.Tensor(args.embedding_size * 2, args.embedding_size * 2))#256*256
        self.u_omega = nn.Parameter(torch.Tensor(args.embedding_size * 2, 1))#256*1
        nn.init.uniform_(self.V, -0.2, 0.2)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        self.state_att = state_attention(args)
        self.dataset = args.dataset
        ##mlm
        if self.dataset == 'cd':
            self.route_mlm_head = nn.Sequential(
                nn.Linear(self.emb_size, 4 * self.emb_size),
                nn.LayerNorm(4 * self.emb_size),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(4 * self.emb_size, 4 * self.emb_size),
                nn.LayerNorm(4 * self.emb_size),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(4 * self.emb_size, token_size_out)
            )
        else:
            self.route_mlm_head = nn.Sequential(
                nn.Linear(self.emb_size, 4 * self.emb_size),
                nn.ReLU(),
                nn.Linear(4 * self.emb_size, token_size_out)
            )

        self.grid_padding_vec = torch.zeros(1, args.embedding_size, requires_grad=True).to(self.device)  # mask_token所在位置的特殊填充
        self.emb_size = args.embedding_size
        self.traj_max_len = args.longest
        self.grid_embed_size = args.grid_embed_size
        self.emb_size_grid = args.emb_size
        self.hidden_size_grid = args.hidden_size_grid
        self.drop_route_rate = args.drop_route_rate
        self.position_embedding1 = nn.Embedding(self.traj_max_len, self.emb_size)  # (100,512)
        self.fc1 = nn.Linear(self.emb_size, self.emb_size)  # route fuse time ffn
        if self.dataset == 'porto':
            self.grid_encoder = TransformerModel(self.emb_size, 8, self.hidden_size_grid, 4, self.drop_route_rate)
        else:
            self.grid_encoder = TransformerModel(self.emb_size, 4, self.hidden_size_grid, 2, self.drop_route_rate)
        self.gps_embed_size = args.gps_embed_size  # 256
        self.gps_embedding = nn.Linear(2, self.gps_embed_size)
        self.position_embedding2 = nn.Embedding(self.traj_max_len, self.gps_embed_size)
        self.gps_encoder = nn.GRU(self.gps_embed_size, self.gps_embed_size, batch_first=True, bidirectional=True)
        self.gps_mlm_head = nn.Linear(self.gps_embed_size*2, 2)
        ####
        # cl project head
        self.hidden_size_cl = args.hidden_size_cl  #128
        self.gps_proj_head = nn.Linear(self.gps_embed_size*2, self.hidden_size_cl)
        self.route_proj_head = nn.Linear(self.emb_size, self.hidden_size_cl)
        # matching
        self.matching_predictor = nn.Linear(self.hidden_size_cl * 2, 2)
        self.register_buffer("gps_queue", torch.randn(self.hidden_size_cl, 2048))
        self.register_buffer("route_queue", torch.randn(self.hidden_size_cl, 2048))

        self.image_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.route_queue, dim=0)

    def scale_process(self, e_inputs, scale_size, lengths):
        e_inputs_split = torch.mean(e_inputs.unfold(1, scale_size, scale_size), dim=3)
        e_inputs_split = self.attention_layer(e_inputs_split, lengths)
        e_inputs_split = pack_padded_sequence(e_inputs_split, lengths, batch_first=True, enforce_sorted=False)
        return e_inputs_split

    def Norm_A(self, A, D):
        return D.mm(A).mm(self.V).mm(D)

    def Norm_A_N(self, A, D):
        return D.mm(A).mm(D)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def padding_mask(self, inp):
        return inp == 0

    def attention_layer(self, e_input, lengths):
        mask = self.getMask(lengths)
        u = torch.tanh(torch.matmul(e_input, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()
        if mask.size(1) != att.size(1):
            mask = mask[:, :att.size(1)] 
        att = att.masked_fill(mask == 0, -1e10)
        att_score = F.softmax(att, dim=1).unsqueeze(2) 
        att_e_input = e_input * att_score  
        return att_e_input

    def array2sparse(self, A):
        A = A.tocoo()
        values = A.data
        indices = np.vstack((A.row, A.col))
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(values).to(self.device)
        A = torch.sparse_coo_tensor(i, v, torch.Size(A.shape), dtype=torch.float32)
        return A

    def getMask(self, seq_lengths):
        max_len = max(seq_lengths)
        mask = torch.ones((len(seq_lengths), max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def encode_grid(self,trajs_grid, masked_grid_assign_mat, mask_token,nodes):
        if masked_grid_assign_mat.dtype != torch.long:
            masked_grid_assign_mat = masked_grid_assign_mat.long()
        nodes = nodes.to(self.device)
        batch_size, max_seq_len = masked_grid_assign_mat.size()
        lookup_table = torch.cat([nodes,self.grid_padding_vec],dim=0).to(self.device)
        grid_emb = torch.index_select(lookup_table, 0,
                                      masked_grid_assign_mat.view(-1))
        grid_emb = grid_emb.view(batch_size, max_seq_len, -1)  
        mask_token = torch.tensor(mask_token, dtype=torch.int64, device=masked_grid_assign_mat.device)
        src_key_padding_mask = (masked_grid_assign_mat == 0)
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)
        position = torch.arange(grid_emb.shape[1]).long().to(self.device)  
        pos_emb = self.position_embedding1(position.unsqueeze(0).repeat(batch_size, 1)) 
        # === 4. Transformer 编码 ===
        grid_emb = grid_emb + pos_emb  # 融合位置编码
        grid_emb = self.fc1(grid_emb)
        grid_enc = self.grid_encoder(grid_emb, None, src_key_padding_mask)  
        grid_enc = torch.where(torch.isnan(grid_enc), torch.full_like(grid_enc, 0), grid_enc)
        grid_unpooled = grid_enc * pool_mask.repeat(1, 1, grid_enc.shape[-1])  # 让 mask 位置变为 0
        grid_pooled = grid_unpooled.sum(1) / pool_mask.sum(1).clamp(min=1)  # (batch_size, embed_size)
        return grid_unpooled, grid_pooled
    
    def encode_gps(self, trajs_gps, masked_gps_assign_mat, mask_token_x, mask_token_y):
        """
        trajs_gps: (batch_size, seq_length, 2)  # 原始GPS坐标
        masked_gps_assign_mat: (batch_size, seq_length, 2)  # 含掩码的GPS坐标
        mask_token_x, mask_token_y: 掩码token值
        """
        if isinstance(trajs_gps, list):
            trajs_gps = torch.tensor(trajs_gps, dtype=torch.float32, device=self.device)
        if isinstance(masked_gps_assign_mat, list):
            masked_gps_assign_mat = torch.tensor(masked_gps_assign_mat, dtype=torch.float32, device=self.device)
        batch_size, seq_length, _ = trajs_gps.shape
        padding_mask = ((masked_gps_assign_mat[:, :, 0] == 0) & (masked_gps_assign_mat[:, :, 1] == 0))
        mask_position_mask = (
                    (masked_gps_assign_mat[:, :, 0] == mask_token_x) & (masked_gps_assign_mat[:, :, 1] == mask_token_y))
        valid_data_mask = (~(padding_mask | mask_position_mask)).float().unsqueeze(-1)
        train_mask = (~padding_mask).float().unsqueeze(-1)
        train_lengths = (train_mask.squeeze(-1).sum(dim=1)).long().clamp(min=1)
        gps_emb = self.gps_embedding(masked_gps_assign_mat)
        pos_indices = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        pos_emb = self.position_embedding2(pos_indices)
        gps_emb = gps_emb + pos_emb
        mask_pos_emb = gps_emb.clone()
        gps_emb = gps_emb * valid_data_mask
        packed_emb = pack_padded_sequence(gps_emb, train_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_encoded, _ = self.gps_encoder(packed_emb)
        gps_encoded, _ = pad_packed_sequence(packed_encoded, batch_first=True, total_length=seq_length)
        gps_unpooled = gps_encoded * train_mask
        valid_sum = (gps_encoded * valid_data_mask).sum(1)
        valid_count = valid_data_mask.sum(1).clamp(min=1)
        gps_pooled = valid_sum / valid_count
        return gps_unpooled, gps_pooled

    def forward(self, trajs,masked_grid,trajs_gps, masked_gps,times, lengths, batch_size, mode, c,mask_token,mask_token_x, mask_token_y):
        adj = self.array2sparse(self.adj)
        d_norm = self.array2sparse(self.d_norm)
        if self.dataset == 'porto':
            H = self.Norm_A(adj, d_norm) 
            nodes = H.mm(self.embedding(self.nodes)).to(self.device)
        else:
            H = self.Norm_A_N(adj, d_norm)
            nodes = H.mm(self.embedding(self.nodes)).mm(self.V).to(self.device)
        s_inputs = torch.index_select(nodes, 0, trajs.flatten().to(self.device)). \
            reshape(batch_size, -1, self.emb_size)
        grid_unpooled, grid_pooled = self.encode_grid(trajs, masked_grid, mask_token, nodes)
        gps_unpooled, gps_pooled = self.encode_gps(trajs_gps, masked_gps, mask_token_x, mask_token_y)
        t_inputs = self.d2v(times.to(self.device)).to(self.device)
        att_s, att_t = self.co_attention(grid_unpooled, t_inputs)
        st_inputs = torch.concat((att_s, att_t), dim=2)
        d_inputs = torch.cat((torch.zeros(batch_size, 1, self.emb_size * 2, dtype=torch.long).to(self.device),
                              st_inputs[:, :-1, :]), dim=1) 
        decoder_inputs = pack_padded_sequence(d_inputs, lengths, batch_first=True, enforce_sorted=False)
        if mode == "pretrain" or "train":
            encoder_inputs_s1 = pack_padded_sequence(self.attention_layer(st_inputs, lengths), lengths,
                                                     batch_first=True, enforce_sorted=False)
            encoder_inputs_s2 = self.scale_process(st_inputs, self.s1_size, [int(i // self.s1_size) for i in lengths])
            encoder_inputs_s3 = self.scale_process(st_inputs, self.s2_size, [int(i // self.s2_size) for i in lengths])
            _, encoder_final_state_s1 = self.encoder_s1(encoder_inputs_s1)
            _, encoder_final_state_s2 = self.encoder_s2(encoder_inputs_s2)
            _, encoder_final_state_s3 = self.encoder_s3(encoder_inputs_s3)
            encoder_final_state = (self.W1 * encoder_final_state_s1 + self.W2 * encoder_final_state_s2
                                   + self.W3 * encoder_final_state_s3)
            sum_W = self.W1.data + self.W2.data + self.W3.data
            self.W1.data /= sum_W
            self.W2.data /= sum_W
            self.W3.data /= sum_W
            mu = self.fc_mu(encoder_final_state)
            logvar = self.fc_logvar(encoder_final_state)
            z = self.reparameterize(mu, logvar)
            decoder_outputs, _ = self.decoder(decoder_inputs, z)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)
        elif mode == "test":
            mu = torch.stack([self.mu_prior] * batch_size, dim=1)[c: c + 1]
            decoder_outputs, _ = self.decoder(decoder_inputs, mu)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)
            logvar, z = None, None
        output = self.fc_out(self.layer_norm(decoder_outputs))
        return output, mu, logvar, z,grid_unpooled, grid_pooled ,gps_unpooled, gps_pooled

