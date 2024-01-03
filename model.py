import torch
from torch import nn
from math import sqrt
import sys
    
    
class Model(nn.Module):

    def __init__(self, vocab_size, N, d_model, d_ff, h, d_k, d_v, P_drop, init_std, positional_encoding_max_pos, masking_minus_inf, device):
        super().__init__()
        
        self.vocab_size=vocab_size
        self.d_model=d_model
        self.device=device
        self.N=N

        # Pre-softmax linear transformation after the decoder; the same transformation transposed produces embeddings out of the original onehot representations        
        self.V=nn.Parameter(torch.empty((d_model, vocab_size), dtype=torch.float32).normal_(std=init_std)) 

        # Learned positional encodings
        self.PE_full=nn.Parameter(torch.empty((positional_encoding_max_pos, d_model), dtype=torch.float32).normal_(std=init_std))
        
        self.embedding_dropout = nn.Dropout(p=P_drop)
        
        self.decoder=Decoder(N, d_model, d_ff, h, d_k, d_v, P_drop, masking_minus_inf, device)
        
        self.final_layer_norm = nn.LayerNorm(self.d_model, device=device)

    def forward(self, input_onehot):
        
        U=torch.transpose(self.V, 0, 1)        
        embedding=torch.matmul(input_onehot,U)
        
        decoder_input=self.embedding_dropout(positional_encoding(embedding, self.PE_full))
                
        decoder_output = self.decoder(decoder_input)
        
        decoder_output_normalized = self.final_layer_norm(decoder_output)
        
        # onehot_pred shape =(batch_size, vocab_size, sent_len)
        onehot_pred=torch.moveaxis(torch.matmul(decoder_output_normalized, self.V), 1, -1)

    
        return onehot_pred
    

class Decoder(nn.Module):
    
    def __init__(self, N, d_model, d_ff, h, d_k, d_v, P_drop, masking_minus_inf, device):
        super().__init__()
                
        self.N=N
        
        for i in range(self.N):
            # if(i==(N-1)):
            #     is_last=True
            # else:
            #     is_last=False
            setattr(self, 'block_' + str(i), Dec_block(d_model, d_ff, h, d_k, d_v, P_drop, masking_minus_inf, device))
                
    def forward(self, decoder_input):
        
        block_output_list=[]
        block_output_list.append(self.block_0(decoder_input))
        for i in range(1,self.N):
            block_output_list.append(getattr(self,'block_' + str(i))(block_output_list[i-1]))
                
        return block_output_list[-1]


class Dec_block(nn.Module):
    
    def __init__(self, d_model, d_ff, h, d_k, d_v, P_drop, masking_minus_inf, device):
        super().__init__()
        
        self.d_model=d_model
        self.mmha=Multi_Head_Attention(d_model, h, d_k, d_v, device, masking_minus_inf=masking_minus_inf, masked=True)
        self.ff=FF(d_model, d_ff, device)
        
        self.layer_norm1 = nn.LayerNorm(self.d_model, device=device)
        self.layer_norm2 = nn.LayerNorm(self.d_model, device=device)
        self.dropout1 = nn.Dropout(p=P_drop)
        self.dropout2 = nn.Dropout(p=P_drop)
        
    def forward(self, block_input):
        
        mmha_input=self.layer_norm1(block_input)
        sublayer2_input = block_input + self.dropout1(self.mmha(mmha_input, mmha_input, mmha_input))
        
        ff_input=self.layer_norm2(sublayer2_input)
        block_output = sublayer2_input + self.dropout2(self.ff(ff_input))
            
        return block_output


class Multi_Head_Attention(nn.Module):
    
    def __init__(self, d_model, h, d_k, d_v, device, masking_minus_inf=None, masked=False):
        super().__init__()
        
        self.h=h
        self.masked=masked
        self.device=device
        self.masking_minus_inf=masking_minus_inf
        
        for i in range(self.h):
            setattr(self, 'Linear_Q_h' + str(i), nn.Linear(in_features=d_model, out_features=d_k, bias=False, device=device))
            setattr(self, 'Linear_K_h' + str(i), nn.Linear(in_features=d_model, out_features=d_k, bias=False, device=device))
            setattr(self, 'Linear_V_h' + str(i), nn.Linear(in_features=d_model, out_features=d_v, bias=False, device=device))
            
        self.Linear_after_concat=nn.Linear(in_features=d_v*h, out_features=d_model, bias=False, device=device)
        
    def forward(self, Q, K, V):
        
        Q_list=[]
        K_list=[]
        V_list=[]
        
        for i in range(self.h):
            Q_list.append(getattr(self,'Linear_Q_h' + str(i))(Q))
            K_list.append(getattr(self,'Linear_K_h' + str(i))(K))
            V_list.append(getattr(self,'Linear_V_h' + str(i))(V))
            
        SDPA_outputs_list=[]
        for i in range(self.h):
            SDPA_outputs_list.append(scaled_dot_product_attention(Q_list[i], K_list[i], V_list[i], self.device, self.masking_minus_inf, masked=self.masked))
            
        concat_output=torch.cat(SDPA_outputs_list, dim=-1)
        
        output=self.Linear_after_concat(concat_output)
            
        return output

def scaled_dot_product_attention(Q, K, V, device, masking_minus_inf, masked):
    
    QK=torch.matmul(Q,torch.transpose(K, -2, -1))/sqrt(K.shape[-1])
    
    if(masked):
        mask_to_add=masking_minus_inf*torch.triu(torch.ones((1, QK.shape[1], QK.shape[2])).to(device), diagonal=1)
        QK_masked=QK+mask_to_add
        # QK_masked=QK  

                
    else:
        QK_masked=QK
    
    output=torch.matmul(nn.functional.softmax(QK_masked, dim=-1),V)
        
    return output

class FF(nn.Module):
    
    def __init__(self, d_model, d_ff, device):
        super().__init__()
        
        self.d_model=d_model
        self.d_ff=d_ff
        
        layers=[]
        layers.append(nn.Linear(in_features=d_model, out_features=d_ff, device=device))
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_features=d_ff, out_features=d_model, device=device))
                
        self.main = nn.Sequential(*layers)
        
    def forward(self, ff_input):
        
        output=self.main(ff_input)
            
        return output


    
def positional_encoding(embedding, PE_full):
    
    PE=PE_full[None,:embedding.shape[1],:]
    
    try:
        res = embedding + PE
    except:
        print(f"Embedding: {embedding.shape}")
        print(f"PE: {PE.shape}")
        sys.exit()
        
    return res

        
    