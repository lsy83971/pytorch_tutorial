import torch
import torch.nn as nn

torch.randn(3, 4)
torch.randn(3, 4) + torch.randn(3, 4) 
a = torch.randn(3, 4)
b = torch.randn(4, 3)

torch.mm(a, b)
torch.matmul(a, b)

# RNN

input_dim = 10
output_dim = 20
num_layer = 2
seq_len = 5
batch_size = 10

rnn = nn.RNN(input_dim, output_dim, num_layer)
input_data = torch.randn(seq_len, batch_size, input_dim)
h0 = torch.randn(num_layer, batch_size, output_dim)
output_data = rnn(input_data, h0)


# nn.MultiheadAttention

embed_dim = 16
num_heads = 1
batch_size = 4
seq_len_query = 10
seq_len_key = 20

## query: (seq_len_query, batch_size ,embed_dim)
## key: (seq_len_key, batch_size ,embed_dim)
## value: (seq_len_key, batch_size ,embed_dim)
query = torch.randn(seq_len_query, batch_size ,embed_dim)
key = torch.randn(seq_len_key, batch_size, embed_dim)
value = torch.randn(seq_len_key, batch_size, embed_dim)
key_padding_mask = (torch.randn(batch_size, seq_len_key) > 0)

multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
attn_output, attn_output_weights = multihead_attn(query, key, value)
linear_m = list(multihead_attn.parameters())[2]
linear_m_input = list(multihead_attn.parameters())[0][32:, ]

#import pdb
#pdb.run("multihead_attn(query, key, value)")

attn_output.shape #[10, 4, 16]
attn_output_weights.shape #[4, 10, 20]


value_permute = value.permute(1, 0, 2)
test_output_at = attn_output_weights@(value_permute@(linear_m_input.T))
test_output_bmm = torch.bmm(attn_output_weights, value_permute)

linear_output1 = nn.functional.linear(test_output_at, linear_m)
linear_output2 = test_output_at@(linear_m)
linear_output3 = test_output_at@(linear_m.T)

linear_output1_final = linear_output1.permute(1, 0, 2)
linear_output2_final = linear_output2.permute(1, 0, 2)

(linear_output1_final - attn_output)

##Great!!!

#################################

## nn.TransformerEncoder
## nn.transformerencoderlayer
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(10, 32, 512)
out = transformer_encoder(src)
# src.shape
# out.shape

for i in encoder_layer.parameters():
    print(i.shape)

for i in transformer_encoder.parameters():
    print(i.shape)


#################################

## nn.TransformerDecoderLayer
nn.TransformerDecoderLayer
nn.TransformerDecoder

decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
memory = torch.rand(10, 32, 512)
tgt = torch.rand(20, 32, 512)
out = transformer_decoder(tgt, memory)

for i in decoder_layer.parameters():
    print(i.shape)




