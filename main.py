import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#word embedding
batch_size = 2

#单词表大小
max_num_src_words = 8
max_num_tgt_words = 8
model_dim = 8

#序列的最大长度
max_src_seq_len = 5
max_tgt_seq_len = 5
max_position_len = 5

#src_len = torch.randint(2,5,(batch_size,))
#tgt_len = torch.randint(2,5,(batch_size,))
src_len = torch.Tensor([2,4]).to(torch.int32)
tgt_len = torch.Tensor([4,3]).to(torch.int32)#int32是PyTorch中跨平台兼容性最好的整数类型(不同设备/框架对int32的支持最稳定)

# 单词索引构成的句子
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_src_words,(L,)),(0,max_src_seq_len-L)),0)
                     for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1,max_num_tgt_words,(L,)),(0,max_tgt_seq_len-L)),0) for L in tgt_len])

#构造word embedding 也就是单词
src_embedding_table = nn.Embedding(max_num_src_words+1,model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words+1,model_dim)
src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq) #对一个实例后面直接加括号就是forward方法

#构造position embedding
pos_mat = torch.arange(max_position_len).reshape((-1,1)) #pos代表行,i代表列
i_mat = torch.pow(10000,torch.arange(0,8,2).reshape((1,-1))/model_dim)
pe_embeding_table = torch.zeros(max_position_len,model_dim)#通过位置索引直接从表中取对应位置的编码（每个位置分配唯一向量），如embedding通过单词索引取向量
pe_embeding_table[:,0::2] = torch.sin(pos_mat/i_mat)
pe_embeding_table[:,1::2] = torch.cos(pos_mat/i_mat)

pe_embeding = nn.Embedding(max_position_len,model_dim)
pe_embeding.weight = nn.Parameter(pe_embeding_table,requires_grad=False)

src_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)),0) for i in range(len(src_len))]).to(torch.int32)
tgt_pos = torch.cat([torch.unsqueeze(torch.arange(max(src_len)),0) for q in src_len]).to(torch.int32)

src_pe_embedding = pe_embeding(src_pos)
tgt_pe_embedding = pe_embeding(tgt_pos)


##构造encoder的self-attention mask
#mask shape [batch_size,max_src_len,max_src_len]
valid_encoder_pos =torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max_src_seq_len-L)),0)
                                              for L in src_len]),2)
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos,valid_encoder_pos.transpose(1,2)) #两个矩阵相乘是计算关联度
invalid_encoder_pos_matrix = 1-valid_encoder_pos_matrix
mask_encoder_self_attention = invalid_encoder_pos_matrix.to(torch.bool)


#构造cross-attention的mask
#Q @ K^T shape:[batch_size,tgt_seq_len,src_seq_len]
#源序列有效位置
valid_encoder_pos =torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max_src_seq_len-L)),0)
                                              for L in src_len]),2)
#目标序列有效位置
valid_decoder_pos =torch.unsqueeze(torch.cat([torch.unsqueeze(F.pad(torch.ones(L),(0,max_tgt_seq_len-L)),0)
                                              for L in tgt_len]),2)
#反映源序列与目标序列的有效性
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos,valid_encoder_pos.transpose(1,2))
invalid_cross_pos_matrix = 1-valid_cross_pos_matrix
mask_cross_self_attention = invalid_cross_pos_matrix.to(torch.bool)

#构造decoder self-attention的mask
valid_decoder_tri_matrix = torch.cat([torch.unsqueeze(F.pad(torch.tril(torch.ones((L,L))),\
                                  (0,max_tgt_seq_len-L,0,max_tgt_seq_len-L)),0) \
                            for L in tgt_len])
invalid_decoder_tri_matrix = 1-valid_decoder_tri_matrix
invalid_decoder_tri_matrix = invalid_decoder_tri_matrix.to(torch.bool)

scores = torch.randn(batch_size,max_tgt_seq_len,max_tgt_seq_len)
masked_score = scores.masked_fill(invalid_decoder_tri_matrix,-np.inf)
prob = F.softmax(masked_score,-1)
print(masked_score,prob)


#构建scaled self-attention
def scaled_dot_attention(Q,K,V,attn_mask):
    scores = torch.bmm(Q,K.transpose(-2,-1))//torch.sqrt(model_dim)
    masked_score = scores.masked_fill(attn_mask,1e-9)
    prob = F.softmax(masked_score,-1)
    context = torch.bmm(prob,V)
    return context
