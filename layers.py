import torch
from torch import nn
from torch.nn import functional as F

global_seed = 0
torch.manual_seed(0)
device = 'cuda:0'

def to_npy(x):
    return x.cpu().data.numpy() if device == 'cuda:0' else x.detach().numpy()
    
class CTRMultiEmbedding(nn.Module):
    def __init__(self, ex_parameters, embedding_size, embedding_layers):
        super(CTRMultiEmbedding, self).__init__()
        self.embedding_t, self.embedding_l, self.embedding_u, self.embedding_su, self.embedding_sl, self.embedding_tu, self.embedding_tl = embedding_layers
        self.ex_su, self.ex_sl, self.ex_tu, self.ex_tl = ex_parameters
        self.embedding_size = embedding_size

    def forward(self, traj_input, mat_input, traj_length):
        # traj_input (N, M, 3), mat_input (N, M, M, 2), traj_length [N]
        hours = 24*7
        traj_input[:, :, 2] = (traj_input[:, :, 2] - 1) % hours + 1  
        time_embedding = self.embedding_t(traj_input[:, :, 2])  
        location_embedding = self.embedding_l(traj_input[:, :, 1])  
        user_embedding = self.embedding_u(traj_input[:, :, 0])  
        joint_embedding = time_embedding + location_embedding + user_embedding  

        delta_spatial, delta_temporal = mat_input[:, :, :, 0], mat_input[:, :, :, 1]
        mask = torch.zeros_like(delta_spatial, dtype=torch.long)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_length[i], 0:traj_length[i]] = 1

        embedding_sl, embedding_su, embedding_tl, embedding_tu = self.embedding_sl(mask), self.embedding_su(mask), self.embedding_tl(mask), self.embedding_tu(mask)
        v_sl, v_su, v_tl, v_tu = (delta_spatial - self.ex_sl).unsqueeze(-1).expand(-1, -1, -1, self.embedding_size), \
                                 (self.ex_su - delta_spatial).unsqueeze(-1).expand(-1, -1, -1, self.embedding_size), \
                                 (delta_temporal - self.ex_tl).unsqueeze(-1).expand(-1, -1, -1, self.embedding_size), \
                                 (self.ex_tu - delta_temporal).unsqueeze(-1).expand(-1, -1, -1, self.embedding_size)

        spatial_interval = (embedding_sl * v_su + embedding_su * v_sl) / (self.ex_su - self.ex_sl)
        temporal_interval = (embedding_tl * v_tu + embedding_tu * v_tl) / (self.ex_tu - self.ex_tl)
        delta_embedding = spatial_interval + temporal_interval  # (N, M, M, embedding)

        return joint_embedding, delta_embedding


class CTRSelfAttention(nn.Module):
    def __init__(self, embedding_size, output_size):
        super(CTRSelfAttention, self).__init__()
        self.query_linear = nn.Linear(embedding_size, output_size, bias=False)
        self.key_linear = nn.Linear(embedding_size, output_size, bias=False)
        self.value_linear = nn.Linear(embedding_size, output_size, bias=False)

    def forward(self, joint_embedding, delta_embedding, traj_length):
        delta_embedding = torch.sum(delta_embedding, -1) 
        # joint_embedding (N, M, emb), delta_embedding (N, M, M, emb), traj_length [N]
        # construct attention mask
        mask = torch.zeros_like(delta_embedding, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_length[i], 0:traj_length[i]] = 1

        attention_scores = torch.add(torch.bmm(self.query_linear(joint_embedding), self.key_linear(joint_embedding).transpose(-1, -2)), delta_embedding)  # (N, M, M)
        attention_scores = F.softmax(attention_scores, dim=-1) * mask  # (N, M, M)

        attention_output = torch.bmm(attention_scores, self.value_linear(joint_embedding))  # (N, M, emb)

        return attention_output  # (N, M, emb)


class CTREmbedding(nn.Module):
    def __init__(self, ex_parameters, embedding_size, max_location, embedding_layers):
        super(CTREmbedding, self).__init__()
        _, _, _, self.embedding_su, self.embedding_sl, self.embedding_tu, self.embedding_tl = embedding_layers
        self.ex_su, self.ex_sl, self.ex_tu, self.ex_tl = ex_parameters
        self.embedding_size = embedding_size
        self.max_location = max_location

    def forward(self, traj_location, mat2, vector, traj_length):
        # traj_location (N, M), mat2 (L, L), vector (N, M), delta_t (N, M, L)
        delta_t = vector.unsqueeze(-1).expand(-1, -1, self.max_location)
        delta_s = torch.zeros_like(delta_t, dtype=torch.float32)
        mask = torch.zeros_like(delta_t, dtype=torch.long)
        for i in range(mask.shape[0]): 
            mask[i, 0:traj_length[i]] = 1
            delta_s[i, :traj_length[i]] = torch.index_select(mat2, 0, (traj_location[i]-1)[:traj_length[i]])

        embedding_sl, embedding_su, embedding_tl, embedding_tu = self.embedding_sl(mask), self.embedding_su(mask), self.embedding_tl(mask), self.embedding_tu(mask)
        v_sl, v_su, v_tl, v_tu = (delta_s - self.ex_sl).unsqueeze(-1).expand(-1, -1, -1, self.embedding_size), \
                                 (self.ex_su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.embedding_size), \
                                 (delta_t - self.ex_tl).unsqueeze(-1).expand(-1, -1, -1, self.embedding_size), \
                                 (self.ex_tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.embedding_size)

        space_interval = (embedding_sl * v_su + embedding_su * v_sl) / (self.ex_su - self.ex_sl)
        time_interval = (embedding_tl * v_tu + embedding_tu * v_tl) / (self.ex_tu - self.ex_tl)
        delta_embedding = space_interval + time_interval  # (N, M, L, embedding)

        return delta_embedding


class SelfAttn_dropout(nn.Module):
    def __init__(self, emb_size, output_size, dropout):
        super(SelfAttn_dropout, self).__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)  # squeeze the embed dimension
        # joint (N, M, emb), delta (N, M, M, emb), len [N]
        # construct attention mask
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)  # (N, M, M)
        attn = F.softmax(attn, dim=-1) * mask  # (N, M, M)

        # Dropout
        attn = self.dropout(attn)
        attn_out = torch.bmm(attn, self.value(joint))  # (N, M, emb)

        return attn_out  # (N, M, emb)


class SelfAttn_MultiHead(nn.Module):
    def __init__(self, emb_size, output_size, num_heads, dropout):
        super(SelfAttn_MultiHead, self).__init__()
        self.num_heads = num_heads

        # Create linear transformations for queries, keys, and values for each head
        self.query_linear = nn.ModuleList([nn.Linear(emb_size, output_size, bias=True) for _ in range(num_heads)])
        self.key_linear = nn.ModuleList([nn.Linear(emb_size, output_size, bias=True) for _ in range(num_heads)])
        self.value_linear = nn.ModuleList([nn.Linear(emb_size, output_size, bias=True) for _ in range(num_heads)])

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Final linear transformation to obtain the output with bias=False
        self.output_linear = nn.Linear(output_size * num_heads, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1)  # Squeeze the embed dimension
        # joint (N, M, emb), delta (N, M, M, emb), len [N]

        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        # Create empty lists to store the results of each head
        attn_heads = []

        for head in range(self.num_heads):
            # Apply linear transformations for queries, keys, and values for each head
            query = self.query_linear[head](joint)
            key = self.key_linear[head](joint)
            value = self.value_linear[head](joint)

            # Calculate attention scores for this head
            attn = torch.bmm(query, key.transpose(-1, -2))
            attn = attn / (joint.size(-1) ** 0.5)  # Scale by sqrt(d_k)

            # Apply mask
            attn = attn * mask

            # Apply softmax to obtain attention weights
            attn = F.softmax(attn, dim=-1)

            # Apply Dropout for regularization
            attn = self.dropout(attn)

            # Apply attention weights to values
            attn_out = torch.bmm(attn, value)

            # Append the output of this head to the list
            attn_heads.append(attn_out)

        # Concatenate the outputs of all heads
        attn_concat = torch.cat(attn_heads, dim=-1)

        # Apply a linear transformation to obtain the final output
        output = self.output_linear(attn_concat)

        return output


class CTRAttention(nn.Module):
    def __init__(self, emb_loc, loc_max, dropout=0.2):
        super(CTRAttention, self).__init__()
        self.value = nn.Linear(100, 1, bias=False)
        self.emb_loc = emb_loc
        self.loc_max = loc_max
    
    def forward(self, self_attn, self_delta, traj_len):
        # self_attn (N, M, emb), candidate (N, L, emb), self_delta (N, M, L, emb), len [N]
        self_delta = torch.sum(self_delta, -1).transpose(-1, -2)  
        [N, L, M] = self_delta.shape
        candidates = torch.linspace(1, int(self.loc_max), int(self.loc_max)).long()  # (L)
        candidates = candidates.unsqueeze(0).expand(N, -1).to(device)  # (N, L)
        emb_candidates = self.emb_loc(candidates)  # (N, L, emb)
        attn = torch.mul(torch.bmm(emb_candidates, self_attn.transpose(-1, -2)), self_delta)  # (N, L, M)
        attn_out = self.value(attn).view(N, L)  # (N, L)
        return attn_out  # (N, L)
