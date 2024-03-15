from layers import *
    
class Model(nn.Module):
    def __init__(self, t_dimension, l_dimension, u_dimension, embedding_dimension, ex, num_heads=2, dropout_rate=0.1):
        super(Model, self).__init__()
        emb_t = nn.Embedding(t_dimension, embedding_dimension, padding_idx=0)
        emb_l = nn.Embedding(l_dimension, embedding_dimension, padding_idx=0)
        emb_u = nn.Embedding(u_dimension, embedding_dimension, padding_idx=0)
        emb_su = nn.Embedding(2, embedding_dimension, padding_idx=0)
        emb_sl = nn.Embedding(2, embedding_dimension, padding_idx=0)
        emb_tu = nn.Embedding(2, embedding_dimension, padding_idx=0)
        emb_tl = nn.Embedding(2, embedding_dimension, padding_idx=0)
        embedding_layers = emb_t, emb_l, emb_u, emb_su, emb_sl, emb_tu, emb_tl

        self.MultiEmbed = CTRMultiEmbedding(ex, embedding_dimension, embedding_layers)
        self.SelfAttention = SelfAttn_MultiHead(embedding_dimension, embedding_dimension, num_heads, dropout_rate)
        self.Embedding = CTREmbedding(ex, embedding_dimension, l_dimension-1, embedding_layers)
        self.Attention = CTRAttention(emb_l, l_dimension-1)

    def forward(self, traj_input, mat1_input, mat2_input, vec_input, traj_length):
        joint, delta = self.MultiEmbed(traj_input, mat1_input, traj_length)  
        self_attn = self.SelfAttention(joint, delta, traj_length)
        self_delta = self.Embedding(traj_input[:, :, 1], mat2_input, vec_input, traj_length) 
        output = self.Attention(self_attn, self_delta, traj_length)  
        return output
