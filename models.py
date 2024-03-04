from layers import *

class Model(nn.Module):
    def __init__(self, t_dim, l_dim, u_dim, embed_dim, ex, num_heads=2, dropout=0.1):
        super(Model, self).__init__()
        emb_t = nn.Embedding(t_dim, embed_dim, padding_idx=0)
        emb_l = nn.Embedding(l_dim, embed_dim, padding_idx=0)
        emb_u = nn.Embedding(u_dim, embed_dim, padding_idx=0)
        emb_su = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_sl = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tu = nn.Embedding(2, embed_dim, padding_idx=0)
        emb_tl = nn.Embedding(2, embed_dim, padding_idx=0)
        embed_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl

        self.MultiEmbed = MultiEmbed(ex, embed_dim, embed_layers)

        # self.SelfAttn = SelfAttn(embed_dim, embed_dim)  # initi
        # self.SelfAttn = SelfAttn_dropout(embed_dim, embed_dim, dropout)
        self.SelfAttn = SelfAttn_MultiHead(embed_dim, embed_dim, num_heads, dropout)

        self.Embed = Embed(ex, embed_dim, l_dim-1, embed_layers)
        self.Attn = Attn(emb_l, l_dim-1)

    def forward(self, traj, mat1, mat2, vec, traj_len):
        joint, delta = self.MultiEmbed(traj, mat1, traj_len)  
        self_attn = self.SelfAttn(joint, delta, traj_len)
        self_delta = self.Embed(traj[:, :, 1], mat2, vec, traj_len) 
        output = self.Attn(self_attn, self_delta, traj_len)  
        return output
