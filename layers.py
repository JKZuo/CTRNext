import torch
from torch import nn
from torch.nn import functional as F

seed = 0
global_seed = 0
hours = 24*7
torch.manual_seed(seed)
device = 'cuda:0'

def to_npy(x):
    return x.cpu().data.numpy() if device == 'cuda:0' else x.detach().numpy()

class MultiEmbed(nn.Module):
    def __init__(self, ex, emb_size, embed_layers):
        super(MultiEmbed, self).__init__()
        self.emb_t, self.emb_l, self.emb_u, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size

    def forward(self, traj, mat, traj_len):
        # traj (N, M, 3), mat (N, M, M, 2), len [N]
        traj[:, :, 2] = (traj[:, :, 2]-1) % hours + 1  
        time = self.emb_t(traj[:, :, 2])  
        loc = self.emb_l(traj[:, :, 1])  
        user = self.emb_u(traj[:, :, 0])  
        joint = time + loc + user  

        delta_s, delta_t = mat[:, :, :, 0], mat[:, :, :, 1]
        mask = torch.zeros_like(delta_s, dtype=torch.long)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl*vsu+esu*vsl) / (self.su-self.sl)
        time_interval = (etl*vtu+etu*vtl) / (self.tu-self.tl)
        delta = space_interval + time_interval  # (N, M, M, emb)

        return joint, delta

class SelfAttn(nn.Module):
    def __init__(self, emb_size, output_size):
        super(SelfAttn, self).__init__()
        self.query = nn.Linear(emb_size, output_size, bias=False)
        self.key = nn.Linear(emb_size, output_size, bias=False)
        self.value = nn.Linear(emb_size, output_size, bias=False)

    def forward(self, joint, delta, traj_len):
        delta = torch.sum(delta, -1) 
        # joint (N, M, emb), delta (N, M, M, emb), len [N]
        # construct attention mask
        mask = torch.zeros_like(delta, dtype=torch.float32)
        for i in range(mask.shape[0]):
            mask[i, 0:traj_len[i], 0:traj_len[i]] = 1

        attn = torch.add(torch.bmm(self.query(joint), self.key(joint).transpose(-1, -2)), delta)  # (N, M, M)
        attn = F.softmax(attn, dim=-1) * mask  # (N, M, M)

        attn_out = torch.bmm(attn, self.value(joint))  # (N, M, emb)

        return attn_out  # (N, M, emb)

class Embed(nn.Module):
    def __init__(self, ex, emb_size, loc_max, embed_layers):
        super(Embed, self).__init__()
        _, _, _, self.emb_su, self.emb_sl, self.emb_tu, self.emb_tl = embed_layers
        self.su, self.sl, self.tu, self.tl = ex
        self.emb_size = emb_size
        self.loc_max = loc_max

    def forward(self, traj_loc, mat2, vec, traj_len):
        # traj_loc (N, M), mat2 (L, L), vec (N, M), delta_t (N, M, L)
        delta_t = vec.unsqueeze(-1).expand(-1, -1, self.loc_max)
        delta_s = torch.zeros_like(delta_t, dtype=torch.float32)
        mask = torch.zeros_like(delta_t, dtype=torch.long)
        for i in range(mask.shape[0]): 
            mask[i, 0:traj_len[i]] = 1
            delta_s[i, :traj_len[i]] = torch.index_select(mat2, 0, (traj_loc[i]-1)[:traj_len[i]])

        esl, esu, etl, etu = self.emb_sl(mask), self.emb_su(mask), self.emb_tl(mask), self.emb_tu(mask)
        vsl, vsu, vtl, vtu = (delta_s - self.sl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.su - delta_s).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (delta_t - self.tl).unsqueeze(-1).expand(-1, -1, -1, self.emb_size), \
                             (self.tu - delta_t).unsqueeze(-1).expand(-1, -1, -1, self.emb_size)

        space_interval = (esl * vsu + esu * vsl) / (self.su - self.sl)
        time_interval = (etl * vtu + etu * vtl) / (self.tu - self.tl)
        delta = space_interval + time_interval  # (N, M, L, emb)

        return delta

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
    
class Attn(nn.Module):
    def __init__(self, emb_loc, loc_max, dropout=0.2):
        super(Attn, self).__init__()
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